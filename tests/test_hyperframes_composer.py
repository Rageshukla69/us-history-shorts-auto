import tempfile
import unittest
from pathlib import Path
import sys
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hyperframes_composer import (
    HyperFramesComposer,
    _extract_smart_stat,
    _build_smart_caption_chunks,
)
from media_fetcher import MediaFetcher
from voice_engine import WordTiming


class HyperFramesComposerTests(unittest.TestCase):
    def test_hyperframes_v10_contract_and_zero_blur(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dummy_img = tmp_path / "dummy.jpg"
            Image.new("RGB", (1080, 1920), (20, 30, 45)).save(dummy_img)
            dummy_wav = tmp_path / "dummy.wav"
            dummy_wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")

            scenes = []
            words = []
            cursor = 0.0
            for i in range(1, 9):
                dur = 3.5
                scenes.append({
                    "scene_id": i,
                    "narration": f"In {1900 + i}, divers recovered 2,000 years old bronze gears.",
                    "visual_subject": f"Scene {i} subject",
                    "on_screen_callout": "OVER 2,000 YEARS OLD" if i == 1 else f"FOUND IN {1900 + i}",
                    "era_tag": f"AEGEAN // {1900 + i}",
                    "asset_path": str(dummy_img),
                    "asset_type": "image",
                    "asset_kind": "AI RECONSTRUCTION (1/10)",
                    "start_sec": cursor,
                    "duration_sec": dur,
                    "end_sec": cursor + dur,
                })
                words.append(WordTiming(word="Recovered", start_sec=cursor + 0.1, end_sec=cursor + 0.5, scene_id=i))
                words.append(WordTiming(word=",", start_sec=cursor + 0.5, end_sec=cursor + 0.62, scene_id=i))
                words.append(WordTiming(word="bronze", start_sec=cursor + 0.58, end_sec=cursor + 1.0, scene_id=i))
                words.append(WordTiming(word="gears.", start_sec=cursor + 1.0, end_sec=cursor + 1.5, scene_id=i))
                cursor += dur

            hf = HyperFramesComposer(project_dir=tmp_path / "hf_proj")
            proj_dir = hf.build_project(
                topic="Antikythera Mechanism",
                title="Ancient Greek Computer #Shorts",
                scenes=scenes,
                word_timings=words,
                master_audio_wav=dummy_wav,
            )
            index_html = (proj_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn('data-composition-id="main"', index_html)
            self.assertIn('data-motion-blur="on"', index_html)
            self.assertIn('window.__timelines["main"] = tl;', index_html)
            self.assertIn('HyperShader.init', index_html)
            self.assertIn('"cinematic-zoom"', index_html)
            self.assertIn('"cross-warp-morph"', index_html)
            self.assertIn('"light-leak"', index_html)
            # Verify ZERO blur filter on scene images
            self.assertNotIn("blur(", index_html)
            # Verify standalone comma was merged onto "RECOVERED," instead of a standalone span
            self.assertIn(">RECOVERED,<", index_html)
            self.assertNotIn(">,<", index_html)
            self.assertTrue((proj_dir / "preview.html").exists())
            self.assertTrue((proj_dir / "DESIGN.md").exists())

    def test_smart_stat_distinguishes_quantities_from_years(self):
        s_qty = {
            "on_screen_callout": "OVER 2,000 YEARS OLD",
            "narration": "A lump of corroded bronze concealed gears more than two millennia old.",
        }
        info_qty = _extract_smart_stat(s_qty)
        self.assertIsNotNone(info_qty)
        self.assertEqual(info_qty["target"], 2000)
        self.assertEqual(info_qty["start"], 0)
        self.assertTrue(info_qty["use_commas"])
        self.assertEqual(info_qty["suffix"], "+")
        self.assertEqual(info_qty["label"], "YEARS OLD")

        s_yr = {
            "on_screen_callout": "FOUND IN 1901",
            "narration": "Sponge divers found it in 1901 among cargo from a shipwreck.",
        }
        info_yr = _extract_smart_stat(s_yr)
        self.assertIsNotNone(info_yr)
        self.assertEqual(info_yr["target"], 1901)
        self.assertFalse(info_yr["use_commas"])
        self.assertEqual(info_yr["suffix"], "")

    def test_caption_chunks_never_overlap_or_strand_punctuation(self):
        raw_words = [
            WordTiming(word="Recovered", start_sec=10.0, end_sec=10.45, scene_id=3),
            WordTiming(word="off", start_sec=10.40, end_sec=10.65, scene_id=3),
            WordTiming(word="Antikythera", start_sec=10.62, end_sec=11.15, scene_id=3),
            WordTiming(word=",", start_sec=11.12, end_sec=11.25, scene_id=3),
            WordTiming(word="the", start_sec=11.20, end_sec=11.40, scene_id=3),
            WordTiming(word="wreck", start_sec=11.38, end_sec=11.75, scene_id=3),
        ]
        chunks = _build_smart_caption_chunks(raw_words)
        self.assertEqual(len(chunks), 2)
        self.assertEqual([w.word for w in chunks[0]], ["Recovered", "off", "Antikythera,"])
        self.assertEqual([w.word for w in chunks[1]], ["the", "wreck"])
        self.assertLess(chunks[0][-1].end_sec, chunks[1][0].start_sec)

    def test_perceptual_dhash_and_schematic_rejection(self):
        # A flat white diagram with 2 solid rectangles should be rejected by _is_valid_photographic_image
        diagram = Image.new("RGB", (600, 600), (250, 250, 250))
        d = ImageDraw.Draw(diagram)
        d.rectangle((100, 100, 500, 300), fill=(255, 0, 255))
        self.assertFalse(MediaFetcher._is_valid_photographic_image(diagram))


if __name__ == "__main__":
    unittest.main()
