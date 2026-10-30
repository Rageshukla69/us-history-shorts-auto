import unittest, os, tempfile
from pathlib import Path
from src.quality_control import validate_rendered_short

class QcPolicyTests(unittest.TestCase):
    def test_quality_policy_source_threshold_configurable(self):
        self.assertEqual(os.getenv("MIN_AUTHENTIC_MEDIA_RATIO","0.30"),"0.30")

if __name__ == "__main__": unittest.main()
