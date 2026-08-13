"""Score the object map against the scene's ground truth, offline.

Why this exists: improving grounding means changing thresholds and merge rules,
and every such change needs the same three numbers per category - how many
objects we ended up with, how many real ones we found, and how many we invented.
Getting them by hand (load the scene-graph dump, load object_list.txt, match by
label, count) is a throwaway script that gets rewritten every time, so it lives
here instead.

No GPU, no simulator, no ROS: it reads the scene-graph snapshot the running node
already writes (config.DEBUG_DIR/scene_graph_latest.json) and the ground-truth
object list straight out of map/<scene>.zip. Runs in seconds, so it is the tool
to iterate against while tuning.

It calls the *production* ObjectMemory.merge_duplicates() and filter_reliable(),
not a copy, so what it reports is what the robot will do.

    # score the latest run (scene auto-detected from the running simulator)
    python3 tests/check_object_memory.py

    # explicit scene, and only the categories a question cares about
    python3 tests/check_object_memory.py --scene home_building_1 --categories sofa,pillow

    # find better thresholds instead of guessing
    python3 tests/check_object_memory.py --sweep

Columns:
  ours       objects in our map for that category
  gt         objects the simulator says exist
  found      GT objects with one of ours within --hit-distance (recall)
  invented   ours further than --fp-distance from any GT of that label (precision)

`found` must never drop when a filter or merge is made stricter - that is the
guard rail. The tool exits non-zero if a stage loses a GT object, so it can gate
a change rather than just describe it.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import subprocess
import sys
import zipfile

import numpy as np

from sysnav import config
from sysnav.memory.object_memory import ObjectMemory, filter_reliable

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DEFAULT_MAP_DIR = os.path.join(REPO_ROOT, "map")
DEFAULT_SNAPSHOT = os.path.join(REPO_ROOT, "ai_module", "debug", "scene_graph_latest.json")
SCENE_LINK = (
    "/home/docker/autonomy_stack_mecanum_wheel_platform/src/base_autonomy/"
    "vehicle_simulator/mesh/unity"
)


def detect_scene() -> str | None:
    """Ask the running simulator which scene is loaded (mesh/unity symlink)."""
    try:
        out = subprocess.run(
            ["docker", "exec", "iros2026_system", "readlink", SCENE_LINK],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    target = out.stdout.strip()
    return os.path.basename(target) if target else None


def load_ground_truth(scene: str, map_dir: str) -> dict[str, list[tuple[float, float, float]]]:
    path = os.path.join(map_dir, f"{scene}.zip")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found - pass --scene / --map-dir")
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith("object_list.txt")]
        if not names:
            raise SystemExit(f"{path} has no object_list.txt")
        text = archive.read(names[0]).decode("utf-8", errors="replace")

    truth: dict[str, list[tuple[float, float, float]]] = {}
    for line in text.splitlines():
        quote = line.find('"')
        if quote < 0:
            continue
        label = line[quote:].strip().strip('"').lower()
        numbers = line[:quote].split()
        if len(numbers) < 4:
            continue
        try:
            position = tuple(float(v) for v in numbers[1:4])
        except ValueError:
            continue
        truth.setdefault(label, []).append(position)  # type: ignore[arg-type]
    return truth


def load_snapshot(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found - run the node once so it writes a scene graph, "
            "or pass --snapshot"
        )
    return json.load(open(path, encoding="utf-8")).get("objects", [])


def build_memory(objects: list[dict]) -> ObjectMemory:
    """Rebuild an ObjectMemory from a snapshot so the production merge/filter can
    run on it. Fields the snapshot does not carry (point clouds, images) are not
    used by either routine, so empty stand-ins are fine."""
    memory = ObjectMemory()
    with memory._lock:
        for item in objects:
            object_id = int(item["object_id"])
            confidence = float(item.get("confidence", 0.0))
            memory._nodes[object_id] = {
                "object_id": object_id,
                "category": str(item.get("category", "?")).lower(),
                "position": tuple(float(v) for v in item["position"]),
                "extent_3d": tuple(float(v) for v in item.get("extent_3d", (0.0, 0.0, 0.0))),
                "bbox_3d_min": tuple(float(v) for v in item.get("bbox_3d_min", (0.0, 0.0, 0.0))),
                "bbox_3d_max": tuple(float(v) for v in item.get("bbox_3d_max", (0.0, 0.0, 0.0))),
                "point_cloud": np.empty((0, 3), np.float32),
                "num_points": int(item.get("num_points", 0)),
                "observation_count": int(item.get("observation_count", 1)),
                "confidence": confidence,
                "representative_confidence": confidence,
                "representative_image": None,
                "context_image": None,
                "first_seen_time": float(item.get("first_seen_time", 0.0)),
                "last_seen_time": float(item.get("last_seen_time", 0.0)),
                "latest_bbox_2d": (0, 0, 1, 1),
                "self_attributes": dict(item.get("self_attributes", {})),
            }
    return memory


def gt_key(category: str, truth: dict) -> str:
    """Our categories come from the question's wording, so they can be plural
    ("pillows") while ground truth is singular ("pillow"). Try both rather than
    silently scoring against an empty GT list."""
    category = category.lower()
    for candidate in (category, category.rstrip("s"), category + "s"):
        if candidate in truth:
            return candidate
    return category


def score(nodes: list[dict], truth_positions: list, hit_distance: float,
          fp_distance: float) -> tuple[int, int]:
    found = sum(
        1 for position in truth_positions
        if min((math.dist(node["position"], position) for node in nodes), default=1e9)
        <= hit_distance
    )
    invented = sum(
        1 for node in nodes
        if min((math.dist(node["position"], position) for position in truth_positions),
               default=1e9) > fp_distance
    )
    return found, invented


def stages(objects: list[dict]) -> dict[str, list[dict]]:
    """The four stages a candidate list can be in, using production routines."""
    raw = build_memory(objects).all_nodes()

    merged_memory = build_memory(objects)
    merged_memory.merge_duplicates()
    merged = merged_memory.all_nodes()

    filtered, _ = filter_reliable(raw)
    both, _ = filter_reliable(merged)
    return {"raw": raw, "merge": merged, "filter": filtered, "merge+filter": both}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", default=None, help="scene name; auto-detected if omitted")
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--map-dir", default=DEFAULT_MAP_DIR)
    parser.add_argument("--categories", default=None,
                        help="comma-separated subset to report (default: every category we mapped)")
    parser.add_argument("--hit-distance", type=float, default=1.0,
                        help="a GT object counts as found within this distance (m)")
    parser.add_argument("--fp-distance", type=float, default=2.0,
                        help="one of ours further than this from any GT counts as invented (m)")
    parser.add_argument("--sweep", action="store_true",
                        help="sweep the filter thresholds instead of reporting one setting")
    args = parser.parse_args()

    scene = args.scene or detect_scene()
    if not scene:
        raise SystemExit("could not detect the loaded scene - pass --scene")
    truth = load_ground_truth(scene, args.map_dir)
    objects = load_snapshot(args.snapshot)
    if not objects:
        raise SystemExit(f"{args.snapshot} has no objects")

    categories = ([c.strip().lower() for c in args.categories.split(",")]
                  if args.categories
                  else sorted({str(o.get("category", "?")).lower() for o in objects}))

    print(f"scene            : {scene}")
    print(f"snapshot         : {args.snapshot} ({len(objects)} objects)")
    print(f"filter in effect : observation_count >= {config.OBJECT_MIN_OBSERVATIONS}, "
          f"confidence >= {config.OBJECT_MIN_CONFIDENCE}")
    print(f"merge in effect  : distance < max({config.OBJECT_MERGE_MIN_DISTANCE_M}, "
          f"size*{config.OBJECT_MERGE_SIZE_RATIO}) or overlap > "
          f"{config.OBJECT_MERGE_OVERLAP_RATIO}")
    print()

    by_stage = stages(objects)

    if args.sweep:
        return sweep(objects, truth, categories, args)

    header = f"{'category':<16}{'stage':<14}{'ours':>6}{'gt':>5}{'found':>7}{'invented':>10}"
    print(header)
    print("-" * len(header))
    regressed = []
    for category in categories:
        key = gt_key(category, truth)
        positions = truth.get(key, [])
        if key != category:
            note = f"  (matched GT label '{key}')"
        else:
            note = ""
        baseline_found = None
        for stage_name in ("raw", "merge", "filter", "merge+filter"):
            nodes = [n for n in by_stage[stage_name] if n["category"] == category]
            found, invented = score(nodes, positions, args.hit_distance, args.fp_distance)
            if baseline_found is None:
                baseline_found = found
            elif found < baseline_found:
                regressed.append((category, stage_name, baseline_found, found))
            print(f"{category if stage_name == 'raw' else '':<16}{stage_name:<14}"
                  f"{len(nodes):>6}{len(positions):>5}{found:>7}{invented:>10}"
                  f"{note if stage_name == 'raw' else ''}")
        print()

    if regressed:
        print("REGRESSION - a stage lost ground-truth objects:")
        for category, stage_name, before, after in regressed:
            print(f"  {category}: {stage_name} found {after}, raw found {before}")
        return 1
    print("no stage lost a ground-truth object.")
    return 0


def sweep(objects: list[dict], truth: dict, categories: list[str], args) -> int:
    """Show what each threshold pair would do. Ranking is deliberately left to the
    reader: 'invented' and 'found' trade off, and which side matters depends on
    the mission (counting punishes invented objects, finding punishes misses)."""
    raw = build_memory(objects).all_nodes()
    merged_memory = build_memory(objects)
    merged_memory.merge_duplicates()
    merged = merged_memory.all_nodes()

    print("sweep over (min observations, min confidence), applied after merging:")
    print(f"{'obs':>5}{'conf':>7}{'kept':>7}{'found':>7}{'invented':>10}")
    original = (config.OBJECT_MIN_OBSERVATIONS, config.OBJECT_MIN_CONFIDENCE)
    try:
        for min_obs in (1, 2, 3, 4, 5):
            for min_conf in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7):
                config.OBJECT_MIN_OBSERVATIONS = min_obs
                config.OBJECT_MIN_CONFIDENCE = min_conf
                kept, _ = filter_reliable(merged)
                total_found = total_invented = total_kept = 0
                for category in categories:
                    positions = truth.get(gt_key(category, truth), [])
                    nodes = [n for n in kept if n["category"] == category]
                    found, invented = score(nodes, positions, args.hit_distance, args.fp_distance)
                    total_found += found
                    total_invented += invented
                    total_kept += len(nodes)
                print(f"{min_obs:>5}{min_conf:>7.2f}{total_kept:>7}{total_found:>7}{total_invented:>10}")
    finally:
        config.OBJECT_MIN_OBSERVATIONS, config.OBJECT_MIN_CONFIDENCE = original

    total_gt = sum(len(truth.get(gt_key(c, truth), [])) for c in categories)
    print(f"\nground truth total for these categories: {total_gt}")
    print("pick the row where 'found' is still at its maximum and 'invented' is lowest;")
    print("a row that raises 'found' is impossible - filtering can only remove.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
