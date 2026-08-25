"""참조 물체를 3D로 못 잡았을 때 관계를 버리지 말고 사진으로 판정한다.

실측 2026-08-25: "go to the picture closest to the door"에서 door가 object_memory에
하나도 없었다(문은 벽과 같은 평면이라 LiDAR로 분리가 안 돼 grounding이 통째로 실패한다).
그러자 _best_effort_step_target이 'nearest door' 제약을 통째로 버리고 "로봇에 가장 가까운
picture"를 골랐다 - 사실상 임의 선택이고, 대시보드에도 그렇게 찍혔다:
    ≈ [stop] picture [nearest door] ('nearest door' NOT applied (door has no 3D position))

좌표가 없어도 후보 **자신의 사진**에 참조 물체가 보이는지는 물어볼 수 있다. 그 코드는
이미 있었는데(relation_image_verifier) selection_job에서만 쓰고 이 폴백에서는 안 썼다.
"""

import unittest

import numpy as np

from sysnav import config
from sysnav.missions import mission3_pipe
from sysnav.task.query_parser import extract_target


def _obj(object_id, category, position, confidence=0.8, context=True):
    return {
        "object_id": object_id,
        "category": category,
        "position": (float(position[0]), float(position[1]), 0.0),
        "confidence": float(confidence),
        "extent_3d": (0.4, 0.4, 0.4),
        "observation_count": 3,
        "self_attributes": {},
        "relation_checks": {},
        "representative_image": None,
        "context_image": np.zeros((4, 4, 3), dtype=np.uint8) if context else None,
    }


class _Memory:
    def __init__(self, objects):
        self._objects = {int(o["object_id"]): o for o in objects}
        self.relation_updates = []

    def find_by_category(self, category):
        wanted = str(category).strip().lower()
        return [dict(o) for o in self._objects.values() if o["category"] == wanted]

    def get(self, object_id):
        found = self._objects.get(int(object_id))
        return dict(found) if found else None

    def all_nodes(self):
        return [dict(o) for o in self._objects.values()]

    def update_relation_checks(self, object_id, checks):
        self.relation_updates.append((int(object_id), checks))


class _Verifier:
    """Gemini 대신 미리 정한 답을 돌려주고, 어느 메서드가 불렸는지 기록한다."""

    def __init__(self, winners=(), raises=False):
        self.winners = set(winners)
        self.raises = raises
        self.calls = []

    def _answer(self, candidates, key):
        return {
            int(c["object_id"]): {key: int(c["object_id"]) in self.winners}
            for c in candidates
        }

    def verify(self, candidates, relation, reference_category):
        self.calls.append(("verify", relation, reference_category))
        if self.raises:
            raise RuntimeError("Gemini 실패")
        return self._answer(candidates, f"verify2|{relation}|{reference_category}")

    def rank_superlative(self, candidates, reference_category, relation="nearest"):
        self.calls.append(("rank", relation, reference_category))
        if self.raises:
            raise RuntimeError("Gemini 실패")
        return self._answer(candidates, f"rank|{relation}|{reference_category}")


class _Logger:
    def info(self, _m): pass
    def warning(self, _m): pass
    def error(self, _m): pass


class _Node:
    def __init__(self, objects, verifier=None):
        self.object_memory = _Memory(objects)
        self.relation_image_verifier = verifier

    def get_logger(self):
        return _Logger()


PICTURES = [
    _obj(1, "picture", (5.0, 0.0), confidence=0.70),   # 로봇에서 멀다 - 정답
    _obj(2, "picture", (0.5, 0.0), confidence=0.90),   # 로봇 바로 앞 - 예전엔 이게 뽑혔다
    _obj(3, "picture", (3.0, 3.0), confidence=0.60),
]
POSE = {"x": 0.0, "y": 0.0, "yaw": 0.0}


def _step():
    return {"resolve": "category", "parsed": extract_target("the picture closest to the door")}


class ResolveByImageTest(unittest.TestCase):
    def test_a_superlative_uses_the_comparison_call(self):
        """"closest to"는 후보별 yes/no로 못 가린다 - 전부 놓고 비교시켜야 한다."""
        verifier = _Verifier(winners=[1])
        node = _Node(PICTURES, verifier)
        picked, basis = mission3_pipe._resolve_reference_by_image(
            node, PICTURES, "nearest", "door", POSE)
        self.assertEqual(int(picked["object_id"]), 1)
        self.assertEqual(verifier.calls, [("rank", "nearest", "door")])
        self.assertIn("image comparison", basis)

    def test_a_plain_relation_uses_the_yes_no_call(self):
        verifier = _Verifier(winners=[3])
        node = _Node(PICTURES, verifier)
        picked, basis = mission3_pipe._resolve_reference_by_image(
            node, PICTURES, "near", "door", POSE)
        self.assertEqual(int(picked["object_id"]), 3)
        self.assertEqual(verifier.calls, [("verify", "near", "door")])
        self.assertIn("image check", basis)

    def test_a_single_candidate_skips_the_comparison(self):
        verifier = _Verifier(winners=[1])
        node = _Node(PICTURES, verifier)
        mission3_pipe._resolve_reference_by_image(node, PICTURES[:1], "nearest", "door", POSE)
        self.assertEqual(verifier.calls[0][0], "verify")

    def test_verdicts_are_cached_into_memory(self):
        """같은 사진에 같은 질문을 두 번 하지 않기 위한 적립."""
        node = _Node(PICTURES, _Verifier(winners=[1]))
        mission3_pipe._resolve_reference_by_image(node, PICTURES, "nearest", "door", POSE)
        self.assertEqual(len(node.object_memory.relation_updates), 3)

    def test_ties_are_broken_by_confidence_not_by_robot_distance(self):
        """로봇 거리로 가르면 재시도마다 목표가 흔들린다(between 쌍 선택과 같은 이유)."""
        node = _Node(PICTURES, _Verifier(winners=[1, 2, 3]))
        picked, _ = mission3_pipe._resolve_reference_by_image(node, PICTURES, "near", "door", POSE)
        self.assertEqual(int(picked["object_id"]), 2)   # confidence 0.90

    def test_nothing_passing_reports_it(self):
        node = _Node(PICTURES, _Verifier(winners=[]))
        picked, basis = mission3_pipe._resolve_reference_by_image(node, PICTURES, "near", "door", POSE)
        self.assertIsNone(picked)
        self.assertIn("no candidate passed", basis)

    def test_a_vlm_error_does_not_propagate(self):
        node = _Node(PICTURES, _Verifier(raises=True))
        picked, basis = mission3_pipe._resolve_reference_by_image(node, PICTURES, "near", "door", POSE)
        self.assertIsNone(picked)
        self.assertIn("failed", basis)

    def test_a_node_without_a_verifier_is_handled(self):
        node = _Node(PICTURES, None)
        picked, _ = mission3_pipe._resolve_reference_by_image(node, PICTURES, "near", "door", POSE)
        self.assertIsNone(picked)


class BestEffortIntegrationTest(unittest.TestCase):
    """_best_effort_step_target이 실제로 이 경로를 타는지."""

    def setUp(self):
        mission3_pipe._REF_LOG_SEEN.clear()

    def test_the_image_fallback_beats_robot_nearest(self):
        """보고된 버그. door가 3D로 없어도 정답 picture를 고른다."""
        node = _Node(PICTURES, _Verifier(winners=[1]))
        position, basis = mission3_pipe._best_effort_step_target(node, _step(), POSE)
        self.assertEqual(position, PICTURES[0]["position"])      # picture#1
        self.assertNotIn("NOT applied", basis)
        self.assertIn("image comparison", basis)

    def test_it_still_degrades_when_the_images_decide_nothing(self):
        node = _Node(PICTURES, _Verifier(winners=[]))
        position, basis = mission3_pipe._best_effort_step_target(node, _step(), POSE)
        self.assertEqual(position, PICTURES[1]["position"])      # 로봇 최근접으로 폴백
        self.assertIn("NOT applied", basis)
        self.assertIn("no candidate passed", basis)              # 왜 그랬는지도 남는다

    def test_a_reference_with_a_3d_position_does_not_call_the_vlm(self):
        """좌표가 있으면 기하로 푸는 게 싸고 정확하다 - 이미지 호출을 낭비하면 안 된다."""
        verifier = _Verifier(winners=[1])
        node = _Node([*PICTURES, _obj(9, "door", (5.2, 0.0))], verifier)
        position, basis = mission3_pipe._best_effort_step_target(node, _step(), POSE)
        self.assertEqual(verifier.calls, [])
        self.assertEqual(position, PICTURES[0]["position"])      # door에 가장 가까운 picture
        self.assertIn("geometric", basis)


class CandidatePruningTest(unittest.TestCase):
    """넓은 맵 대비 - 이미지 30장을 한 번에 비교시키면 VLM이 무너진다."""

    @staticmethod
    def _many(count):
        # x가 클수록 로봇(0,0)에서 멀다. id는 일부러 거리와 반대 순서로 매겨서
        # "id 순"이 아니라 "거리 순"으로 잘리는지 확인한다.
        return [
            _obj(count - index, "picture", (float(index) + 1.0, 0.0), confidence=0.5)
            for index in range(count)
        ]

    def test_only_the_cap_is_sent_to_the_vlm(self):
        verifier = _Verifier(winners=[])
        node = _Node(self._many(20), verifier)
        mission3_pipe._resolve_reference_by_image(
            node, self._many(20), "nearest", "door", POSE)
        self.assertEqual(len(node.object_memory.relation_updates),
                         config.RELATION_IMAGE_MAX_CANDIDATES)

    def test_the_nearest_ones_survive(self):
        captured = {}

        class _Capturing(_Verifier):
            def rank_superlative(self, candidates, reference_category, relation="nearest"):
                captured["ids"] = [int(c["object_id"]) for c in candidates]
                return super().rank_superlative(candidates, reference_category, relation)

        node = _Node(self._many(20), _Capturing(winners=[]))
        mission3_pipe._resolve_reference_by_image(
            node, self._many(20), "nearest", "door", POSE)
        # 거리 1.0m..8.0m가 살아남고, id는 20..13이다(거리와 반대로 매긴 id).
        self.assertEqual(captured["ids"], list(range(20, 20 - config.RELATION_IMAGE_MAX_CANDIDATES, -1)))

    def test_the_same_input_gives_the_same_selection(self):
        """캐시가 걸리려면 잘린 집합이 결정적이어야 한다."""
        runs = []
        for _ in range(2):
            captured = {}

            class _Capturing(_Verifier):
                def verify(self, candidates, relation, reference_category):
                    captured["ids"] = sorted(int(c["object_id"]) for c in candidates)
                    return super().verify(candidates, relation, reference_category)

            node = _Node(self._many(20), _Capturing(winners=[]))
            mission3_pipe._resolve_reference_by_image(
                node, self._many(20), "near", "door", POSE)
            runs.append(captured["ids"])
        self.assertEqual(runs[0], runs[1])

    def test_a_small_candidate_set_is_untouched(self):
        verifier = _Verifier(winners=[1])
        node = _Node(PICTURES, verifier)
        _, basis = mission3_pipe._resolve_reference_by_image(
            node, PICTURES, "nearest", "door", POSE)
        self.assertNotIn("top", basis)

    def test_the_basis_records_that_it_was_pruned(self):
        node = _Node(self._many(20), _Verifier(winners=[20]))
        _, basis = mission3_pipe._resolve_reference_by_image(
            node, self._many(20), "nearest", "door", POSE)
        self.assertIn(f"top {config.RELATION_IMAGE_MAX_CANDIDATES} of 20", basis)


if __name__ == "__main__":
    unittest.main()
