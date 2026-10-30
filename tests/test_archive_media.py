import unittest
from src.archive_media import rights_allowed, rank_candidates

class ArchiveRightsTests(unittest.TestCase):
    def test_public_domain_allowed(self): self.assertTrue(rights_allowed("Public Domain"))
    def test_cc_by_allowed(self): self.assertTrue(rights_allowed("CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/"))
    def test_openverse_short_license_code(self): self.assertTrue(rights_allowed("by", "https://creativecommons.org/licenses/by/4.0/"))
    def test_cc_by_sa_allowed(self): self.assertTrue(rights_allowed("CC BY-SA 4.0"))
    def test_noncommercial_rejected(self): self.assertFalse(rights_allowed("CC BY-NC 4.0"))
    def test_no_derivatives_rejected(self): self.assertFalse(rights_allowed("CC BY-ND 4.0"))
    def test_unknown_rejected(self): self.assertFalse(rights_allowed(""))
    def test_loc_no_known_restrictions_statement(self): self.assertTrue(rights_allowed("No known restrictions on publication"))
    def test_scene_relevance_ranking(self):
        scene={"visual_subject":"B-17 bomber aircraft", "visual_keywords":["B-17","bomber","aircraft"]}
        c=[{"title":"B-17 bomber aircraft in flight","desc":"World War II","media_type":"image"},{"title":"New York street","desc":"City life","media_type":"image"}]
        ranked=rank_candidates(c,scene)
        self.assertGreater(ranked[0]["relevance_score"],ranked[1]["relevance_score"])

if __name__ == "__main__": unittest.main()
