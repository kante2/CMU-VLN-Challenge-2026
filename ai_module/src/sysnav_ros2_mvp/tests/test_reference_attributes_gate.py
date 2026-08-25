""""go near the lamp closest to the **black** chair"에서 검은 의자를 아직 못 봤을 때.

실측 2026-08-25: 흰 식탁의자 4개만 보이는 상태에서 로봇이 엉뚱한 lamp로 주행했다.
경로는 이랬다 - selection_job은 reference_allowed_ids로 chair를 black으로 걸러서
통과하는 후보가 0이 되자 relation_pending을 냈다(여기까진 옳다). 그런데
_missing_categories가 "chair 카테고리가 존재하는가"만 봐서 흰 의자 4개를 관측 완료로
처리했고, 폴백인 _best_effort_step_target은 참조에 속성 필터를 아예 안 걸어서 흰 의자를
기준으로 lamp를 골랐다.

즉 "아직 못 본 참조"와 "봤지만 속성이 안 맞는 참조"가 구분되지 않았다.
"""

import unittest

from sysnav.missions import mission3_pipe
from sysnav.task.query_parser import extract_target


def _obj(object_id, category, position, attributes=None):
    return {
        "object_id": object_id,
        "category": category,
        "position": (float(position[0]), float(position[1]), 0.0),
        "confidence": 0.8,
        "extent_3d": (0.4, 0.4, 0.4),
        "observation_count": 3,
        "self_attributes": dict(attributes or {}),
        "representative_image": None,
    }


class _Memory:
    def __init__(self, objects):
        self._objects = {int(o["object_id"]): o for o in objects}

    def find_by_category(self, category):
        wanted = str(category).strip().lower()
        return [dict(o) for o in self._objects.values() if o["category"] == wanted]

    def get(self, object_id):
        found = self._objects.get(int(object_id))
        return dict(found) if found else None

    def all_nodes(self):
        return [dict(o) for o in self._objects.values()]

    def update_self_attributes(self, object_id, attributes):
        self._objects[int(object_id)]["self_attributes"].update(attributes)


class _Verifier:
    """self_attributes에 이미 들어있는 판정만 돌려준다(=캐시 적중). 캐시에 없는 것은
    응답에서 빼서 fail-closed 동작을 그대로 재현한다."""

    def verify(self, candidates, attributes):
        return {
            int(c["object_id"]): {
                a: c["self_attributes"][a] for a in attributes if a in c["self_attributes"]
            }
            for c in candidates
        }


class _Logger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass

    def error(self, _message):
        pass


class _Node:
    def __init__(self, objects):
        self.object_memory = _Memory(objects)
        self.attribute_verifier = _Verifier()

    def get_logger(self):
        return _Logger()


def _step():
    return {
        "resolve": "category",
        "parsed": extract_target("the lamp closest to the black chair"),
    }


WHITE_CHAIRS = [
    _obj(10, "chair", (2.0, 0.0), {"black": False}),
    _obj(11, "chair", (2.5, 0.0), {"black": False}),
]
LAMP_NEAR_WHITE = _obj(3, "lamp", (2.2, 1.0))
LAMP_NEAR_BLACK = _obj(4, "lamp", (-5.0, 0.0))
BLACK_CHAIR = _obj(20, "chair", (-5.2, 0.0), {"black": True})


class MissingCategoriesTest(unittest.TestCase):
    def test_white_chairs_do_not_satisfy_a_black_chair_reference(self):
        """이 가드가 통과하면 폴백이 흰 의자를 기준으로 목적지를 정해버린다."""
        node = _Node([LAMP_NEAR_WHITE, *WHITE_CHAIRS])
        missing = mission3_pipe._missing_categories(node, _step())
        self.assertTrue(any("chair" in entry for entry in missing), missing)
        self.assertTrue(any("black" in entry for entry in missing), missing)

    def test_a_verified_black_chair_satisfies_the_reference(self):
        node = _Node([LAMP_NEAR_WHITE, LAMP_NEAR_BLACK, *WHITE_CHAIRS, BLACK_CHAIR])
        self.assertEqual(mission3_pipe._missing_categories(node, _step()), [])

    def test_an_unverified_chair_is_still_missing(self):
        """fail-closed: 아직 판정 안 된 의자는 "black일 수도 있다"로 통과시키지 않는다."""
        node = _Node([LAMP_NEAR_WHITE, _obj(12, "chair", (2.0, 0.0))])
        self.assertTrue(mission3_pipe._missing_categories(node, _step()))

    def test_a_category_never_observed_is_still_reported(self):
        node = _Node([*WHITE_CHAIRS])  # lamp가 아예 없다
        self.assertIn("lamp", mission3_pipe._missing_categories(node, _step()))

    def test_a_step_without_attribute_constraints_is_unchanged(self):
        step = {"resolve": "category", "parsed": extract_target("the lamp near the chair")}
        node = _Node([LAMP_NEAR_WHITE, *WHITE_CHAIRS])
        self.assertEqual(mission3_pipe._missing_categories(node, step), [])


class BestEffortReferenceAttributeTest(unittest.TestCase):
    POSE = {"x": 0.0, "y": 0.0, "yaw": 0.0}

    def test_the_black_chair_is_used_as_the_reference_when_available(self):
        node = _Node([LAMP_NEAR_WHITE, LAMP_NEAR_BLACK, *WHITE_CHAIRS, BLACK_CHAIR])
        position, basis = mission3_pipe._best_effort_step_target(node, _step(), self.POSE)
        self.assertEqual(position, LAMP_NEAR_BLACK["position"])
        self.assertNotIn("NOT applied", basis)

    def test_degrading_to_unfiltered_references_is_reported(self):
        """탐사가 소진돼 폴백이 불가피할 때도 "black을 무시했다"가 로그에 남아야 한다."""
        node = _Node([LAMP_NEAR_WHITE, LAMP_NEAR_BLACK, *WHITE_CHAIRS])
        position, basis = mission3_pipe._best_effort_step_target(node, _step(), self.POSE)
        self.assertEqual(position, LAMP_NEAR_WHITE["position"])
        self.assertIn("NOT applied", basis)
        self.assertIn("black", basis)


if __name__ == "__main__":
    unittest.main()
