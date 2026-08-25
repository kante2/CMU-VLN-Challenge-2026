"""테스트 때 사용 허용된 시스템 토픽만 구독한다 + /way_point 제거의 주행 대체.

README의 "System Outputs" 표에 6개 토픽이 있고, 바로 아래에 이렇게 명시돼 있다:
  "While more topics may be available from the system, these are the only ones
   allowed to be used during test time."

그런데 SysNav는 base autonomy(waypointConverter)가 발행하는 /way_point를 추가로
구독하고 있었다. 우리가 발행하는 것은 /way_point_with_heading이고 /way_point는 그것을
시스템이 travArea 기준으로 **갈아끼운 결과**라, 표 밖의 시스템 출력을 읽는 위반이었다.

/way_point는 진단 외에 실제 주행에도 쓰였다: mission3의 도착 판정 기준(target_goal_xy)을
base autonomy가 확정한 좌표로 사후 동기화했다. 그게 없으면 publish()가 스냅해서 보낸
좌표 B에 로봇이 도착해도 원본 A까지 거리가 남아 같은 subgoal을 재발행하다 포기한다.
이제 그 동기화를 **발행 시점에** 우리가 직접 한다(_publish_target_goal).
"""

import ast
import pathlib
import unittest

SYSNAV = pathlib.Path(__file__).resolve().parents[1] / "sysnav"

# README "System Outputs" 표 + 별도로 명시된 질문 토픽.
ALLOWED_SYSTEM_TOPICS = {
    "/camera/image",
    "/registered_scan",
    "/sensor_scan",
    "/terrain_map",
    "/terrain_map_ext",
    "/state_estimation",
    "/challenge_question",
}


def _topic_constants() -> dict[str, str]:
    """config.py의 TOPIC_* 상수를 이름 -> 값으로."""
    tree = ast.parse((SYSNAV / "config.py").read_text())
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (isinstance(target, ast.Name) and target.id.startswith("TOPIC_")
                    and isinstance(node.value, ast.Constant)):
                found[target.id] = node.value.value
    return found


def _subscribed_topic_constants() -> set[str]:
    """sysnav_node.py의 create_subscription(...)에 넘긴 config.TOPIC_* 이름들."""
    tree = ast.parse((SYSNAV / "sysnav_node.py").read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "create_subscription"):
            continue
        for argument in node.args:
            if (isinstance(argument, ast.Attribute)
                    and isinstance(argument.value, ast.Name)
                    and argument.value.id == "config"):
                names.add(argument.attr)
    return names


class AllowedTopicsTest(unittest.TestCase):
    def test_every_subscription_is_an_allowed_topic(self):
        constants = _topic_constants()
        subscribed = _subscribed_topic_constants()
        self.assertTrue(subscribed, "구독을 하나도 못 찾았다 - 파서가 깨진 것")
        for name in sorted(subscribed):
            with self.subTest(constant=name):
                topic = constants.get(name)
                self.assertIsNotNone(topic, f"{name}이 config.py에 없다")
                self.assertIn(
                    topic, ALLOWED_SYSTEM_TOPICS,
                    f"{name}={topic}은 README System Outputs 표에 없는 토픽이다",
                )

    def test_way_point_is_not_referenced_anywhere_in_the_runtime(self):
        """/way_point_with_heading(우리 출력)은 되지만 /way_point(시스템 출력)는 안 된다."""
        offenders = []
        for path in SYSNAV.rglob("*.py"):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                stripped = line.split("#", 1)[0]
                if '"/way_point"' in stripped or "'/way_point'" in stripped:
                    offenders.append(f"{path.name}:{number}")
        self.assertEqual(offenders, [], f"/way_point 참조가 남아있다: {offenders}")

    def test_the_allowed_set_matches_the_readme_table(self):
        """표가 바뀌면 이 테스트부터 깨지도록 README를 실제로 읽어 대조한다."""
        readme = (SYSNAV.parents[3] / "README.md").read_text()
        table = readme.split("#### System Outputs", 1)[1].split("**IMPORTANT NOTE**", 1)[0]
        for topic in ("/camera/image", "/registered_scan", "/sensor_scan",
                      "/terrain_map", "/terrain_map_ext", "/state_estimation"):
            self.assertIn(f"`{topic}`", table, f"{topic}이 README 표에서 사라졌다")
        self.assertNotIn("`/way_point`", table)


class TargetGoalSyncTest(unittest.TestCase):
    """/way_point 동기화를 대체한 발행-시점 동기화가 실제로 코드에 있는지."""

    def test_publish_target_goal_syncs_the_arrival_reference(self):
        source = (SYSNAV / "sysnav_node.py").read_text()
        body = source.split("def _publish_target_goal", 1)[1].split("\n    def ", 1)[0]
        self.assertIn("x, y = float(published.x), float(published.y)", body)
        self.assertIn("if is_final:", body)
        self.assertIn("self.target_goal_xy = (x, y)", body)

    def test_intermediate_hops_do_not_move_the_arrival_reference(self):
        """경유 hop까지 target_goal_xy를 옮기면 최종 목적지를 잃는다."""
        source = (SYSNAV / "sysnav_node.py").read_text()
        body = source.split("def _publish_target_goal", 1)[1].split("\n    def ", 1)[0]
        sync_line = body.index("self.target_goal_xy = (x, y)")
        guard_line = body.index("if is_final:")
        self.assertLess(guard_line, sync_line, "동기화가 is_final 가드 안에 있어야 한다")


if __name__ == "__main__":
    unittest.main()
