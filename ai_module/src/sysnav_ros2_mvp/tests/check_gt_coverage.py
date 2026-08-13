"""Score a finished exploration run against the scene's ground-truth floor.

The coverage percentage the live checker prints while running is
self-referential: its denominator is "floor discovered so far", not the real
room. A region never mapped at all (behind a door, never faced) is simply
absent from that denominator, so a run can report ~100% while an entire area
was never seen. This tool supplies the missing denominator using
`traversable_area.ply` from the scene zip - the simulator's own ground truth
for where the robot can go.

Usage (on the host, from ai_module/src/sysnav_ros2_mvp):
    python3 tests/check_gt_coverage.py hotel_room_1

It reads ../../debug/live_exploration_result.npz (written by
tests/test_live_exploration.py at the end of a run) plus
../../../map/<scene>.zip, and reports:

  observed / true      - how much of the REAL floor the robot saw up close
  mapped / true        - how much of the REAL floor even made it onto the map
  free outside truth   - mapped free cells that are not real floor at all
                         (phantom space; a large number means walls were
                         erased from the map - see the wall-erasure note in
                         tests/test_room_exploration.py)

It also writes a comparison PNG next to the npz.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DEFAULT_NPZ = os.path.join(REPO_ROOT, "ai_module", "debug", "live_exploration_result.npz")
DEFAULT_MAP_DIR = os.path.join(REPO_ROOT, "map")

OCC_FREE = 0
OCC_OCCUPIED = 100


def read_traversable_xy(zip_path: str, scene: str) -> np.ndarray:
    """Return the ground-truth traversable points as an (N, 2) array of x, y."""
    member = f"{scene}/traversable_area.ply"
    with zipfile.ZipFile(zip_path) as archive:
        try:
            raw = archive.read(member)
        except KeyError:
            candidates = [n for n in archive.namelist() if n.endswith("traversable_area.ply")]
            if not candidates:
                raise SystemExit(f"{zip_path} has no traversable_area.ply")
            raw = archive.read(candidates[0])

    text = raw.decode("utf-8", errors="replace")
    header_end = text.find("end_header")
    if header_end == -1:
        raise SystemExit("unexpected PLY: no end_header (binary PLY is not supported here)")
    header = text[:header_end]
    if "format ascii" not in header:
        raise SystemExit("unexpected PLY: only ascii format is supported")

    body = text[header_end + len("end_header"):].strip().splitlines()
    points = []
    for line in body:
        parts = line.split()
        if len(parts) >= 2:
            try:
                points.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if not points:
        raise SystemExit("traversable_area.ply contained no parsable points")
    return np.asarray(points, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scene", help="scene name, e.g. hotel_room_1 (matches map/<scene>.zip)")
    parser.add_argument("--npz", default=DEFAULT_NPZ, help=f"default: {DEFAULT_NPZ}")
    parser.add_argument("--map-dir", default=DEFAULT_MAP_DIR, help=f"default: {DEFAULT_MAP_DIR}")
    args = parser.parse_args()

    if not os.path.exists(args.npz):
        raise SystemExit(
            f"{args.npz} not found - run tests/test_live_exploration.py first "
            "(it writes this file when the run ends)"
        )
    zip_path = os.path.join(args.map_dir, f"{args.scene}.zip")
    if not os.path.exists(zip_path):
        raise SystemExit(f"{zip_path} not found")

    state = np.load(args.npz)
    grid = state["grid"]
    observed = state["observed"]
    origin_x = float(state["origin_x"])
    origin_y = float(state["origin_y"])
    resolution = float(state["resolution"])
    cell_area = resolution ** 2

    truth_xy = read_traversable_xy(zip_path, args.scene)
    cols = np.floor((truth_xy[:, 0] - origin_x) / resolution).astype(np.int64)
    rows = np.floor((truth_xy[:, 1] - origin_y) / resolution).astype(np.int64)
    inside = (rows >= 0) & (rows < grid.shape[0]) & (cols >= 0) & (cols < grid.shape[1])
    dropped = int((~inside).sum())
    truth_mask = np.zeros(grid.shape, dtype=bool)
    truth_mask[rows[inside], cols[inside]] = True

    truth_cells = int(truth_mask.sum())
    if truth_cells == 0:
        raise SystemExit(
            "no ground-truth cell landed inside the mapped grid - the scene and the run "
            "probably do not match, or the world frames differ"
        )

    free = grid == OCC_FREE
    observed_true = int((observed & truth_mask).sum())
    mapped_true = int((free & truth_mask).sum())
    free_outside = int((free & ~truth_mask).sum())
    observed_outside = int((observed & ~truth_mask).sum())

    print(f"scene                : {args.scene}")
    print(f"grid resolution      : {resolution:.2f} m/cell")
    print(f"ground-truth floor   : {truth_cells} cells ({truth_cells * cell_area:.1f} m2)"
          + (f"  [{dropped} GT points fell outside the grid]" if dropped else ""))
    print()
    print(f"observed / true      : {observed_true / truth_cells:6.1%}  "
          f"({observed_true}/{truth_cells} cells, "
          f"{(truth_cells - observed_true) * cell_area:.1f} m2 never seen up close)")
    print(f"mapped   / true      : {mapped_true / truth_cells:6.1%}  "
          f"({mapped_true}/{truth_cells} cells) "
          "<- how much of the real floor reached the map at all")
    print(f"free outside truth   : {free_outside} cells ({free_outside * cell_area:.1f} m2)"
          " <- phantom free space (0 is ideal)")
    print(f"observed outside     : {observed_outside} cells "
          "<- robot 'saw' space that is not real floor")

    self_reported = (
        int((observed & free).sum()) / int(free.sum()) if int(free.sum()) else 0.0
    )
    print()
    print(f"self-reported number : {self_reported:6.1%}  (observed / mapped-free, what the "
          "live run prints)")
    print(f"true number          : {observed_true / truth_cells:6.1%}  (observed / ground truth)")

    try:
        import cv2
    except ImportError:
        return 0

    img = np.zeros((*grid.shape, 3), dtype=np.uint8)
    img[truth_mask] = (60, 60, 60)                      # real floor, not mapped
    img[truth_mask & free] = (170, 200, 255)            # real floor, mapped, not observed
    img[truth_mask & observed] = (255, 255, 255)        # real floor, observed  (the goal)
    img[free & ~truth_mask] = (0, 0, 220)               # phantom free space (red)
    img[grid == OCC_OCCUPIED] = (0, 0, 0)
    known = np.argwhere(truth_mask | (grid != -1))
    if len(known):
        (r0, c0), (r1, c1) = known.min(axis=0), known.max(axis=0)
        pad = 6
        img = img[max(0, r0 - pad):r1 + pad, max(0, c0 - pad):c1 + pad]
    if img.size:
        img = cv2.resize(img, (img.shape[1] * 4, img.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
        out = os.path.join(os.path.dirname(args.npz), f"gt_coverage_{args.scene}.png")
        cv2.imwrite(out, img)
        print()
        print(f"comparison image     : {out}")
        print("  white=observed real floor, pale blue=mapped but not approached, "
              "dark gray=real floor never mapped, red=phantom free space, black=obstacle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
