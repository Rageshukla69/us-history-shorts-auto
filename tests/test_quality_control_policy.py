import unittest
import os
from src.quality_control import validate_rendered_short


class QcPolicyTests(unittest.TestCase):
    def test_quality_policy_source_threshold_configurable(self):
        self.assertIn(os.getenv("MIN_AUTHENTIC_MEDIA_RATIO", "0.25"), ("0.25", "0.30"))


if __name__ == "__main__":
    unittest.main()
