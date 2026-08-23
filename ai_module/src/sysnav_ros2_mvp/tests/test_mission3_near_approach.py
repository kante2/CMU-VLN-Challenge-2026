"""Mission 3 point constraints choose the correct navigation semantics."""

import unittest

from sysnav.missions.mission3_pipe import _uses_object_approach


class Mission3NearApproachTest(unittest.TestCase):
    def test_near_object_uses_dynamic_terrain_approach(self):
        self.assertTrue(_uses_object_approach("near"))

    def test_between_keeps_the_geometric_constraint_point(self):
        self.assertFalse(_uses_object_approach("between"))
        self.assertFalse(_uses_object_approach("between_collective"))


if __name__ == "__main__":
    unittest.main()
