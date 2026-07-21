import unittest

from sysnav.memory.object_association import association_metrics


class AssociationTest(unittest.TestCase):
    def test_near_same_category(self):
        existing = {
            "category": "chair",
            "position": (1.0, 2.0, 0.5),
            "extent_3d": (0.5, 0.5, 1.0),
            "representative_image": None,
        }
        observation = {
            "category": "chair",
            "position": (1.1, 2.05, 0.52),
            "extent_3d": (0.52, 0.48, 1.02),
            "crop_image": None,
        }
        result = association_metrics(existing, observation)
        self.assertTrue(result["allowed"])
        self.assertGreater(result["score"], 0.58)

    def test_different_category(self):
        result = association_metrics(
            {"category": "chair", "position": (0, 0, 0), "extent_3d": (1, 1, 1)},
            {"category": "table", "position": (0, 0, 0), "extent_3d": (1, 1, 1)},
        )
        self.assertFalse(result["allowed"])


if __name__ == "__main__":
    unittest.main()
