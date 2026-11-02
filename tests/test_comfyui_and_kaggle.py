import os
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from comfyui_client import ComfyUIVideoClient
from kaggle_comfy_bridge import is_running_on_kaggle, auto_disconnect_kaggle_gpu


class ComfyUIAndKaggleTests(unittest.TestCase):
    def test_ltx_i2v_workflow_structure(self):
        client = ComfyUIVideoClient(api_url="http://127.0.0.1:8188")
        wf = client._build_ltx_i2v_prompt("scene_01_ai.jpg", "Slow dolly in over bronze gears", num_frames=73)
        self.assertEqual(wf["1"]["class_type"], "CheckpointLoaderSimple")
        self.assertEqual(wf["7"]["class_type"], "LTXVImgToVideo")
        self.assertEqual(wf["7"]["inputs"]["width"], 512)
        self.assertEqual(wf["7"]["inputs"]["height"], 896)
        self.assertEqual(wf["7"]["inputs"]["length"], 73)
        self.assertEqual(wf["15"]["class_type"], "SaveAnimatedWEBP")

    def test_svd_xt_workflow_structure(self):
        client = ComfyUIVideoClient(api_url="http://127.0.0.1:8188")
        wf = client._build_svd_xt_prompt("scene_01_ai.jpg", num_frames=25)
        self.assertEqual(wf["1"]["class_type"], "ImageOnlyCheckpointLoader")
        self.assertEqual(wf["3"]["class_type"], "SVD_img2vid_Conditioning")
        self.assertEqual(wf["7"]["class_type"], "SaveAnimatedWEBP")

    def test_graceful_skip_when_comfyui_offline(self):
        client = ComfyUIVideoClient(api_url="")
        self.assertFalse(client.is_server_reachable())
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            res = client.animate_keyframe_to_video(p / "in.jpg", "motion", 4.0, p / "out.mp4")
            self.assertIsNone(res)

    def test_auto_disconnect_noop_outside_kaggle(self):
        # Outside Kaggle, auto_disconnect_kaggle_gpu should cleanly return without exiting the test runner
        auto_disconnect_kaggle_gpu(comfy_proc=None, force_kernel_exit=False)


if __name__ == "__main__":
    unittest.main()
