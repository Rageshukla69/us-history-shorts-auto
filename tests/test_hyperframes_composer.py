import tempfile
import unittest
from pathlib import Path
import sys
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hyperframes_composer import HyperFramesComposer
from voice_engine import WordTiming


class HyperFramesComposerTests(unittest.TestCase):
    def test_hyperframes_html_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dummy_img = tmp_path / "dummy.jpg"
            Image.new("RGB", (1080, 1920), (20, 30, 45)).save(dummy_img)
            dummy_wav = tmp_path / "dummy.wav"
            dummy_wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")

            scenes = []
            words = []
            cursor = 0.0
            for i in range(1, 7):
                dur = 3.0
                scenes.append({
                    "scene_id": i,
                    "narration": f"In {1940 + i} soldiers deployed secret tanks.",
                    "visual_subject": f"Scene {i} subject",
                    "on_screen_callout": f"JUNE {1940 + i}",
                    "asset_path": str(dummy_img),
                    "asset_type": "image",
                    "asset_kind": "AI RECONSTRUCTION (1/10)",
                    "start_sec": cursor,
                    "duration_sec": dur,
                    "end_sec": cursor + dur,
                })
                words.append(WordTiming(word="Secret", start_sec=cursor + 0.1, end_sec=cursor + 0.6, scene_id=i))
                words.append(WordTiming(word=str(1940 + i), start_sec=cursor + 0.6, end_sec=cursor + 1.2, scene_id=i))
                cursor += dur

            hf = HyperFramesComposer(project_dir=tmp_path / "hf_proj")
            proj_dir = hf.build_project(
                topic="Ghost Army WWII",
                title="The Ghost Army #Shorts",
                scenes=scenes,
                word_timings=words,
                master_audio_wav=dummy_wav,
            )
            index_html = (proj_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn('data-composition-id="main"', index_html)
            self.assertIn('data-width="1080"', index_html)
            self.assertIn('data-height="1920"', index_html)
            self.assertIn('window.__timelines["main"] = tl;', index_html)
            self.assertIn('HyperShader.init', index_html)
            self.assertTrue((proj_dir / "preview.html").exists())
            self.assertTrue((proj_dir / "DESIGN.md").exists())


if __name__ == "__main__":
    unittest.main()
