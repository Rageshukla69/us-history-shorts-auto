import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huashu_art_motion import HuashuArtMotionDirector
from hyperframes_composer import HyperFramesComposer
from voice_engine import WordTiming
from PIL import Image


class HuashuArtMotionTests(unittest.TestCase):
    def test_camera_choreography_never_pushes_three_times_in_a_row(self):
        director = HuashuArtMotionDirector()
        dummy_scenes = [{"scene_id": i} for i in range(1, 11)]
        moves = director.plan_camera_choreography(dummy_scenes)
        self.assertEqual(len(moves), 10)
        self.assertEqual(moves[-1], "resolution_pull_back")
        push_set = {"snap_push", "diagonal_push", "slow_push_in"}
        consec = 0
        for m in moves:
            if m in push_set:
                consec += 1
            else:
                consec = 0
            self.assertLess(consec, 3, f"Found 3 consecutive push-ins in {moves}")

    def test_grammar_assignment_covers_vox_map_data_orbital_and_living_canvas(self):
        director = HuashuArtMotionDirector()
        scenes = [
            {"scene_id": 1, "narration": "A secret treaty letter in the archive.", "visual_subject": "Treaty document"},
            {"scene_id": 2, "narration": "They sailed across the Aegean map route.", "visual_subject": "Naval route map"},
            {"scene_id": 3, "narration": "Over 2,000 years old with 30 bronze gears.", "stat_value": "2,000+"},
            {"scene_id": 4, "narration": "The solar eclipse orbit predicted the moon phase.", "visual_subject": "Solar eclipse"},
            {"scene_id": 5, "narration": "Soldiers stood on the riverbank at dusk.", "visual_subject": "Ancient riverbank"},
        ]
        lint = director.lint_storyboard("The Eclipse That Stopped a War", scenes, ["585 BCE", "Ancient Greece"])
        self.assertEqual(lint["status"], "GREEN")
        grammars = [s["huashu_grammar"] for s in scenes]
        self.assertEqual(grammars[0], "y2_vox_desk")
        self.assertEqual(grammars[1], "y3_whiteboard_map")
        self.assertEqual(grammars[2], "t3_data_chronicle")
        self.assertEqual(grammars[3], "y1_kurzgesagt_scale")

    def test_hyperframes_embeds_huashu_layers_with_zero_blur_and_zero_snow(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            img_path = tmp_path / "sample.jpg"
            Image.new("RGB", (1080, 1920), color=(40, 32, 24)).save(img_path)
            wav_path = tmp_path / "audio.wav"
            wav_path.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")

            scenes = []
            words = []
            for i in range(1, 9):
                scenes.append({
                    "scene_id": i,
                    "narration": f"Scene {i} historical beat with map and archive.",
                    "visual_subject": f"Subject {i}",
                    "duration_sec": 3.5,
                    "asset_path": str(img_path),
                    "asset_type": "image",
                    "on_screen_callout": f"BEAT {i}",
                    "era_tag": "585 BCE",
                    "stat_value": "585" if i == 3 else "",
                })
                words.append(WordTiming(word="HISTORY", start_sec=(i - 1) * 3.5 + 0.2, end_sec=(i - 1) * 3.5 + 0.8, scene_id=i))

            comp = HyperFramesComposer(project_dir=tmp_path / "hf_proj")
            proj_dir = comp.build_project(
                topic="The Eclipse That Stopped a War",
                title="The Eclipse That Stopped a War #Shorts",
                scenes=scenes,
                word_timings=words,
                master_audio_wav=wav_path,
            )
            html_str = (proj_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("huashu-layer", html_str)
            self.assertIn("data-huashu-grammar=", html_str)
            self.assertIn("data-huashu-camera=", html_str)
            self.assertNotIn("blur(", html_str)


if __name__ == "__main__":
    unittest.main()
