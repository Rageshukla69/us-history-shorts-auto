import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huashu_art_motion import HuashuArtMotionDirector
from hyperframes_composer import HyperFramesComposer, _extract_smart_stat
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

    def test_year_1807_next_to_rabbits_never_formats_with_comma_or_redundant_counter(self):
        # Exact reproduction of Scene 1 from 1000047695.mp4
        scene1 = {
            "stat_value": "1807",
            "stat_label": "REPORTED YEAR",
            "era_tag": "FRANCE // 1807",
            "on_screen_callout": "RABBITS NOT SOLDIERS",
            "narration": "A famous 1807 story says Napoleon fled a hunt, not from soldiers, but rabbits.",
        }
        # Because 1807 is already in era_tag ("FRANCE // 1807"), _extract_smart_stat returns None
        self.assertIsNone(_extract_smart_stat(scene1))

        # Even if era_tag does not contain 1807, use_commas MUST be False
        scene_yr = {
            "stat_value": "1807",
            "stat_label": "REPORTED YEAR",
            "era_tag": "FRENCH EMPIRE",
            "on_screen_callout": "RABBITS NOT SOLDIERS",
            "narration": "A famous 1807 story says Napoleon fled a hunt.",
        }
        st_yr = _extract_smart_stat(scene_yr)
        self.assertIsNotNone(st_yr)
        self.assertFalse(st_yr["use_commas"])
        self.assertEqual(st_yr["target"], 1807)

    def test_stage_assignments_include_vox_print_data_split_letterbox_and_fullbleed(self):
        director = HuashuArtMotionDirector()
        scenes = [
            {"scene_id": 1, "narration": "Hook scene.", "era_tag": "FRANCE // 1807", "on_screen_callout": "RABBITS NOT SOLDIERS"},
            {"scene_id": 2, "narration": "Marshal Berthier planned the hunt.", "era_tag": "FRENCH EMPIRE", "on_screen_callout": "BERTHIER PLANNED IT"},
            {"scene_id": 3, "narration": "Over 3,000 rabbits were gathered.", "stat_value": "3,000", "stat_label": "FAMOUS COUNT", "on_screen_callout": "THOUSANDS GATHERED"},
            {"scene_id": 4, "narration": "Tame farm rabbits charged across the field.", "on_screen_callout": "TAME FARM RABBITS"},
            {"scene_id": 5, "narration": "Napoleon retreated to his carriage.", "on_screen_callout": "CARRIAGE RETREAT"},
        ]
        lint = director.lint_storyboard("Napoleon Rabbit Hunt", scenes, ["1807", "France"], stat_extractor=_extract_smart_stat)
        self.assertEqual(lint["status"], "GREEN")
        grammars = [s["huashu_grammar"] for s in scenes]
        self.assertEqual(grammars[0], "fullbleed_hook_stage")
        self.assertEqual(grammars[1], "vox_print_stage")
        self.assertEqual(grammars[2], "data_split_stage")
        self.assertEqual(grammars[3], "cinema_letterbox_stage")
        self.assertEqual(grammars[4], "fullbleed_hook_stage")

    def test_zero_duplicate_callout_text_and_zero_pointer_lines_in_generated_html(self):
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
                    "narration": f"Scene {i} historical beat.",
                    "visual_subject": f"Subject {i}",
                    "duration_sec": 3.5,
                    "asset_path": str(img_path),
                    "asset_type": "image",
                    "on_screen_callout": f"UNIQUE CALLOUT {i}",
                    "era_tag": f"ERA TAG {i}",
                    "stat_value": "3,000" if i == 3 else "",
                    "stat_label": "FAMOUS COUNT" if i == 3 else "",
                })
                words.append(WordTiming(word="HISTORY", start_sec=(i - 1) * 3.5 + 0.2, end_sec=(i - 1) * 3.5 + 0.8, scene_id=i))

            comp = HyperFramesComposer(project_dir=tmp_path / "hf_proj")
            proj_dir = comp.build_project(
                topic="Napoleon Rabbit Hunt",
                title="Napoleon Rabbit Hunt #Shorts",
                scenes=scenes,
                word_timings=words,
                master_audio_wav=wav_path,
            )
            html_str = (proj_dir / "index.html").read_text(encoding="utf-8")
            # 1. Verify each unique callout string appears EXACTLY ONCE in the entire HTML
            for i in range(1, 9):
                self.assertEqual(html_str.count(f"UNIQUE CALLOUT {i}"), 1, f"UNIQUE CALLOUT {i} appeared more than once!")
            # 2. Verify zero blind pointer lines / pins / debug labels
            self.assertNotIn("vox-string", html_str)
            self.assertNotIn("vox-pin", html_str)
            self.assertNotIn("HUASHU // VOX INVESTIGATION DESK", html_str)
            self.assertNotIn("blur(", html_str)
            # 3. Verify new Huashu stages are present
            self.assertIn("vox-print-card", html_str)
            self.assertIn("huashu-data-card", html_str)
            self.assertIn("letterbox-top", html_str)


if __name__ == "__main__":
    unittest.main()
