"""Mission 3는 참조 물체까지 다 본 뒤에도 VLM 확정을 기다리며 서 있으면 안 된다.

"go to the toilet and go to the bedside table near the window."의 2번째 step처럼
관계가 붙은 step은 selection_job이 relation/attribute/verification pending을 돌려줄 수
있다. 그런데 selection_job의 유일한 탈출구인 `mission2_exploration_deadline_reached`는
Mission 2 전용이라 Mission 3에서는 절대 서지 않는다 - 예전 코드는 pending을 받으면
무조건 PLAN_EXPLORATION으로 되돌렸고, 관계가 끝내 검증 안 되면(참조가 유리창이라 3D
grounding이 안 되는 등) 타겟도 참조도 이미 눈앞에 있는데 탐사가 소진될 때까지 돌다가
FAILED로 끝났다.

Mission 3의 채점은 subgoal 순서와 실제 주행 궤적이고 부분점수가 있으므로, 필요한
카테고리가 전부 관측됐다면 기하로 목적지를 정해 **바로** goal을 찍는 것이 항상 낫다.
"""

import unittest
from threading import RLock

from sysnav.missions import mission3_pipe


class _Logger:
    def info(self, _message): pass
    def warning(self, _message): pass
    def error(self, _message): pass


class _ObjectMemory:
    def __init__(self, nodes):
        self._nodes = nodes

    def find_by_category(self, category):
        return [dict(node) for node in self._nodes if node["category"] == category]

    def get(self, object_id):
        for node in self._nodes:
            if node["object_id"] == int(object_id):
                return dict(node)
        return None


class _Node:
    def __init__(self, objects, steps, exhausted=False):
        self.state_lock = RLock()
        self.sensor_lock = RLock()
        self.state = "MISSION3_SELECT_STEP"
        self.mission3_step_index = 0
        self.mission3_exploration_exhausted = exhausted
        self.mission3_forbidden_mask = None
        self.mission3_gate_segment = None
        self.mission3_gate_crossed = False
        self.mission3_gate_last_xy = None
        self.mission3_gate_last_stamp = 0.0
        self.object_memory = _ObjectMemory(objects)
        self.latest_pose = {"x": 0.0, "y": 0.0, "yaw": 0.0, "stamp": 0.0}
        self.task = {"steps": steps}
        self.navigations = []

    # _start_navigate_to_point이 부르는 것들
    def start_target_navigation(self, pose, goal_xy, final_theta, **kwargs):
        self.navigations.append((goal_xy, kwargs))

    def approach_pose_for(self, _pose, position, **_kwargs):
        return float(position[0]), float(position[1]), 0.0

    def refresh_goal_marker(self):
        pass

    def get_logger(self):
        return _Logger()


def _object(object_id, category, x, y):
    return {
        "object_id": object_id,
        "category": category,
        "position": (x, y, 0.5),
        "self_attributes": {},
    }


def _relation_step(target="bedside table", reference="window"):
    """"go to the bedside table near the window."에 해당하는 step."""
    return {
        "resolve": "category",
        "is_stop": True,
        "parsed": {
            "target": target,
            "attributes": [],
            "relation_chain": [(target, "near", reference)],
            "detection_prompts": [target, reference],
        },
    }


class PendingCommitTest(unittest.TestCase):
    def _run(self, node, pending="relation_pending"):
        mission3_pipe._resolve_pending_step(node, {pending: True})

    def test_commits_to_the_candidate_that_satisfies_the_relation(self):
        """창문에 더 가까운 bedside table을 바로 고른다 - 탐사로 안 돌아간다."""
        node = _Node(
            objects=[
                _object(1, "bedside table", 5.0, 0.0),   # 창문에서 5m
                _object(2, "bedside table", 0.5, 0.0),   # 창문에서 0.5m <- 정답
                _object(3, "window", 0.0, 0.0),
            ],
            steps=[_relation_step()],
        )
        self._run(node)
        self.assertEqual(len(node.navigations), 1, "바로 goal을 찍어야 한다")
        self.assertEqual(node.navigations[0][0], (0.5, 0.0))
        self.assertNotEqual(node.state, "PLAN_EXPLORATION")

    def test_farthest_picks_the_other_end(self):
        step = _relation_step()
        step["parsed"]["relation_chain"] = [("bedside table", "farthest", "window")]
        node = _Node(
            objects=[
                _object(1, "bedside table", 5.0, 0.0),
                _object(2, "bedside table", 0.5, 0.0),
                _object(3, "window", 0.0, 0.0),
            ],
            steps=[step],
        )
        self._run(node)
        self.assertEqual(node.navigations[0][0], (5.0, 0.0))

    def test_still_explores_while_a_referenced_object_is_unseen(self):
        """창문을 아직 못 봤으면 예전대로 탐사를 계속한다 - 여기서 성급히 확정하면
        관계를 아예 못 본 채로 찍는 것이라 의미가 없다."""
        node = _Node(
            objects=[_object(1, "bedside table", 5.0, 0.0)],
            steps=[_relation_step()],
        )
        self._run(node)
        self.assertEqual(node.state, "PLAN_EXPLORATION")
        self.assertEqual(node.navigations, [])

    def test_exhausted_exploration_commits_even_with_something_unseen(self):
        """더 볼 곳이 없으면 증거가 모자라도 진행한다(부분점수 > 아무것도 안 함)."""
        node = _Node(
            objects=[_object(1, "bedside table", 5.0, 0.0)],
            steps=[_relation_step()],
            exhausted=True,
        )
        self._run(node)
        self.assertEqual(node.navigations[0][0], (5.0, 0.0))

    def test_every_pending_kind_takes_the_same_path(self):
        for pending in ("relation_pending", "attribute_pending", "verification_pending"):
            with self.subTest(pending=pending):
                node = _Node(
                    objects=[
                        _object(1, "bedside table", 0.5, 0.0),
                        _object(2, "window", 0.0, 0.0),
                    ],
                    steps=[_relation_step()],
                )
                self._run(node, pending)
                self.assertEqual(len(node.navigations), 1)

    def test_no_target_candidate_falls_back_to_exploration(self):
        node = _Node(
            objects=[_object(1, "window", 0.0, 0.0)],
            steps=[_relation_step()],
            exhausted=True,
        )
        self._run(node)
        self.assertEqual(node.state, "PLAN_EXPLORATION")
        self.assertEqual(node.navigations, [])


class BestEffortTargetTest(unittest.TestCase):
    POSE = {"x": 0.0, "y": 0.0, "yaw": 0.0, "stamp": 0.0}

    def test_no_relation_uses_the_nearest_candidate_to_the_robot(self):
        step = {
            "resolve": "category",
            "parsed": {"target": "toilet", "attributes": [], "detection_prompts": ["toilet"]},
        }
        node = _Node(
            objects=[_object(1, "toilet", 9.0, 0.0), _object(2, "toilet", 2.0, 0.0)],
            steps=[step],
        )
        position, basis = mission3_pipe._best_effort_step_target(node, step, self.POSE)
        self.assertEqual(position, (2.0, 0.0, 0.5))
        self.assertIn("no relation", basis)

    def test_reference_without_a_position_is_reported_as_unapplied(self):
        """참조 물체가 object_memory에 아예 없으면 관계를 못 건다 - 로그가 그 사실을
        분명히 말해야 한다(금지구역 미적용 로그와 같은 원칙)."""
        node = _Node(
            objects=[_object(1, "bedside table", 3.0, 0.0)],
            steps=[_relation_step()],
        )
        step = node.task["steps"][0]
        position, basis = mission3_pipe._best_effort_step_target(node, step, self.POSE)
        self.assertEqual(position, (3.0, 0.0, 0.5))
        self.assertIn("NOT applied", basis)

    def test_the_closest_of_several_references_is_used(self):
        """창문이 여러 개면 그중 가장 가까운 것과의 거리로 후보를 고른다."""
        node = _Node(
            objects=[
                _object(1, "bedside table", 0.0, 4.0),
                _object(2, "bedside table", 0.0, 9.0),
                _object(3, "window", 0.0, 0.0),
                _object(4, "window", 0.0, 10.0),   # 2번 바로 옆
            ],
            steps=[_relation_step()],
        )
        step = node.task["steps"][0]
        position, _ = mission3_pipe._best_effort_step_target(node, step, self.POSE)
        self.assertEqual(position, (0.0, 9.0, 0.5))


if __name__ == "__main__":
    unittest.main()
