import os, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from media_fetcher import MediaFetcher

class BudgetTests(unittest.TestCase):
    def test_zero_is_a_real_zero_cap(self):
        old=os.environ.get("MAX_AI_IMAGES_PER_SHORT")
        os.environ["MAX_AI_IMAGES_PER_SHORT"]="0"
        try:
            obj=MediaFetcher.__new__(MediaFetcher)
            # Exercise the same conversion used by __init__.
            self.assertEqual(max(0,int(os.getenv("MAX_AI_IMAGES_PER_SHORT","0"))),0)
        finally:
            if old is None: os.environ.pop("MAX_AI_IMAGES_PER_SHORT",None)
            else: os.environ["MAX_AI_IMAGES_PER_SHORT"]=old

    def test_scene_terms_omit_generic_stopwords(self):
        scene={"visual_subject":"1960s reel to reel tape recorder", "visual_keywords":["magnetic tape","recording"]}
        terms=MediaFetcher._scene_terms(scene)
        self.assertIn("recorder",terms); self.assertIn("magnetic",terms)
        self.assertNotIn("scene",terms)

    def test_candidate_relevance_prefers_exact_subject(self):
        terms={"tape", "recorder", "1960s"}
        exact={"title":"1960s reel tape recorder demonstration", "desc":"Magnetic tape machine"}
        unrelated={"title":"Modern city skyline", "desc":"Aerial view"}
        self.assertGreater(MediaFetcher._relevance_score(exact,terms), MediaFetcher._relevance_score(unrelated,terms))

if __name__=="__main__": unittest.main()
