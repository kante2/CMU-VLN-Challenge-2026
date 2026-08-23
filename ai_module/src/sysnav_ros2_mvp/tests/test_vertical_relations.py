"""under / above 는 높이차만이 아니라 수평 정렬도 봐야 한다.

발견(2026-08-23, map/livingroom_1.zip의 object_list.txt와 대조):
`under` 판정이 높이차만 보고 수평 거리를 전혀 안 봐서, 방 반대편 물체끼리 성립했다.

    cabinet#1(2.19,-0.79) --under--> picture#8(-2.62,-6.55)   conf=0.937  수평 7.51m

그리고 이 엉터리 edge가 "vase on the cabinet below the picture"의 체인을 완성해,
GT 정답(vase#23, (-2.39,-7.11))이 아니라 방 반대편 화병을 가리키게 만들었다.
정답 체인은 두 번째 hop이 0.396으로 임계값 0.55에 못 미쳐 오히려 탈락했다.
"""

import sys
import types
import unittest

import numpy as np

for name in ("cv2",):
    if name not in sys.modules:
        try:
            __import__(name)
        except ImportError:                       # pragma: no cover
            sys.modules[name] = types.ModuleType(name)

from sysnav import config                                             # noqa: E402
from sysnav.reasoning.spatial_relation_reasoner import (              # noqa: E402
    SpatialRelationReasoner as Reasoner,
)


def _box(cx, cy, cz, sx, sy, sz):
    half = np.array([sx, sy, sz], dtype=np.float64) / 2.0
    center = np.array([cx, cy, cz], dtype=np.float64)
    return center - half, center + half


def _under(source, target):
    """source --under--> target 판정."""
    smin, smax = source
    tmin, tmax = target
    return Reasoner._vertical_relation(float(tmin[2] - smax[2]), smin, smax, tmin, tmax)


# 실측 좌표 (2026-08-23 scene_graph_latest.json)
CABINET_NEAR = _box(-2.23, -6.25, 0.48, 0.40, 1.60, 0.90)   # GT cabinet#2에 대응
PICTURE      = _box(-2.62, -6.55, 1.86, 0.05, 0.90, 0.95)   # GT picture#88에 대응
CABINET_FAR  = _box(2.19, -0.79, 0.99, 0.50, 0.60, 1.70)    # 방 반대편


class HorizontalGateTest(unittest.TestCase):
    def test_object_across_the_room_is_not_under(self):
        holds, confidence, reason = _under(CABINET_FAR, PICTURE)
        self.assertFalse(holds, "7.5m 떨어진 물체가 '아래'일 수 없다")
        self.assertEqual(confidence, 0.0)
        self.assertIn("xy_gap", reason, "수평 거리가 근거에 남아야 한다")

    def test_the_real_pair_holds_above_the_selection_threshold(self):
        holds, confidence, _ = _under(CABINET_NEAR, PICTURE)
        self.assertTrue(holds)
        self.assertGreaterEqual(
            confidence, config.SCENE_GRAPH_RELATION_MIN_CONFIDENCE,
            "정답 쌍이 임계값 아래로 떨어지면 선택에서 탈락한다",
        )

    def test_the_real_pair_beats_the_far_one(self):
        self.assertGreater(_under(CABINET_NEAR, PICTURE)[1], _under(CABINET_FAR, PICTURE)[1])


class VerticalGateTest(unittest.TestCase):
    def test_wrong_vertical_order_fails(self):
        holds, _, _ = _under(PICTURE, CABINET_NEAR)     # 그림이 캐비닛 "아래"?
        self.assertFalse(holds)

    def test_normal_picture_height_is_not_penalised_into_rejection(self):
        """그림은 캐비닛에 붙어 있지 않고 0.5~1.2m 위에 걸린다. 그걸 감점하면
        정상 쌍이 탈락한다(수정 전 곱셈 방식에서 0.501 < 0.55로 떨어졌다)."""
        cabinet = _box(0.0, 0.0, 0.45, 0.40, 1.60, 0.90)
        picture = _box(0.0, 0.0, 1.70, 0.05, 0.90, 0.95)   # 높이차 약 0.35m
        holds, confidence, _ = _under(cabinet, picture)
        self.assertTrue(holds)
        self.assertGreater(confidence, 0.8, "완벽히 정렬된 쌍은 높은 점수여야 한다")

    def test_ceiling_far_above_is_rejected(self):
        cabinet = _box(0.0, 0.0, 0.45, 0.40, 1.60, 0.90)
        ceiling = _box(0.0, 0.0, 3.60, 2.00, 2.00, 0.10)
        holds, _, _ = _under(cabinet, ceiling)
        self.assertFalse(holds, f"{config.SCENE_GRAPH_VERTICAL_MAX_GAP_M}m 넘게 뜨면 '아래'가 아니다")


class XYGapTest(unittest.TestCase):
    def test_overlapping_boxes_report_zero_gap(self):
        a = _box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        b = _box(0.2, 0.2, 5.0, 1.0, 1.0, 1.0)
        self.assertEqual(Reasoner._xy_gap(a[0], a[1], b[0], b[1]), 0.0)

    def test_size_mismatch_does_not_break_alignment(self):
        """넓은 캐비닛(1.6m) 위의 좁은 그림(0.9m)은 중심이 어긋나도 겹친다.
        중심 거리로 판정하면 이 정상 쌍을 놓친다."""
        wide = _box(0.0, 0.0, 0.45, 0.40, 1.60, 0.90)
        narrow = _box(0.0, 0.55, 1.70, 0.05, 0.90, 0.95)     # 중심이 0.55m 어긋남
        self.assertEqual(Reasoner._xy_gap(wide[0], wide[1], narrow[0], narrow[1]), 0.0)
        self.assertTrue(_under(wide, narrow)[0])


if __name__ == "__main__":
    unittest.main()
