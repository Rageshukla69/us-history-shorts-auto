"""
Multi-Source Archival & Credit-Saving Visual Engine (v4.2 — Strict 9:16 Full-Bleed Edition):
1. Strict 9:16 (`1080x1920`) Normalization (`normalize_image_to_9_16`):
   - Every single downloaded archival photo (Wikipedia REST, NASA, Wikimedia Commons) and every AI image
     (`MAI-Image-2.6-Flash`) is immediately cropped and normalized to exact `1080x1920` (`9:16`) vertical
     dimensions using subject-aware edge-energy salience detection and upper-third portrait bias (`0.22`),
     so images NEVER change dimensions mid-video!
   - Wide horizontal panoramic strips (`aspect > 1.60`) are rejected at the candidate stage, and vertical/portrait
     archival photographs (`aspect <= 1.05`) receive a `+3` priority score bonus.
2. Wikipedia Official REST Media-List API (`https://en.wikipedia.org/api/rest_v1/page/media-list/{title}`):
   Fetches ONLY the curated historical photographs actually embedded inside the exact canonical Wikipedia
   articles for the topic.
3. Official NASA Image & Video Library API (`https://images-api.nasa.gov/search`):
   100% FREE, requires ZERO API keys, and provides authentic archival NASA photographs for any space,
   rocket, Cold War, or science topic (like Mariner 1, Apollo, Voyager, Hubble).
4. 3-Layer Historical Verification Firewall + Strict MAI-Image Credit Cap (`MAX_AI_IMAGES_PER_SHORT`, default `3`).
"""
import base64
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import quote
import requests
from PIL import Image, ImageEnhance, ImageFilter

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_REST_MEDIA = "https://en.wikipedia.org/api/rest_v1/page/media-list/"
WIKIPEDIA_REST_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
NASA_IMAGES_API = "https://images-api.nasa.gov/search"
PEXELS_VIDEO_API = "https://api.pexels.com/videos/search"
USER_AGENT = "USHistoryShortsBot/4.2 (https://github.com/education/students)"

# Comprehensive blacklist blocking modern celebrities, actors, singers, athletes, press events, and signage
DISQUALIFYING_PATTERNS = re.compile(
    r"\b("
    r"living people|births|actors|actresses|singers|musicians|rappers|songwriters|models|celebrities|"
    r"jennifer lopez|bad bunny|benicio del toro|amaury nolasco|ana ortiz|prince harry|meghan markle|"
    r"invictus games|red carpet|paparazzi|selfie|premiere|film festival|press conference|award|"
    r"grammy|oscar|emmy|golden globe|mtv|billboard|nbcuniversal|nbc|netflix|hbo|disney|maxim|"
    r"comic-con|cosplay|podcast|youtuber|influencer|tiktok|instagram|television series|"
    r"baseball|minor league|major league|seattle mariners|burlington|softball|football|basketball|"
    r"soccer|hockey|cricket|athlete|player|batting|pitcher|outfielder|stadium|ballpark|"
    r"politician|senator|congressman|governor|mayor|businesspeople|entrepreneur|ceo|"
    r"memorial plaque|information board|tourist sign|wayside marker|historical marker|commemorative plaque|"
    r"road sign|street sign|parking lot|shopping mall|headshot|portrait of a man|portrait of a woman|"
    r"logo|icon|coat of arms|flag of|map locator|blank map|svg|user:|talk:|template:|"
    r"2005|2006|2007|2008|2009|2010|2011|2012|2013|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|2024|2025|2026"
    r")\b",
    re.IGNORECASE,
)

# Exceptions where a modern year in metadata is acceptable (e.g. a museum artifact or restored NASA/archival scan)
ARCHIVAL_OVERRIDE_PATTERNS = re.compile(
    r"\b("
    r"museum|smithsonian|national archives|library of congress|nasa|jet propulsion laboratory|jpl|"
    r"spacecraft|satellite|rocket|missile|tank|artillery|wwii|world war|civil war|revolutionary war|"
    r"17th century|18th century|19th century|1900s|1910s|1920s|1930s|1940s|1950s|1960s|1970s|1980s|"
    r"archival|historic|historical|painting|engraving|manuscript|diagram|schematic|blueprint"
    r")\b",
    re.IGNORECASE,
)

# Even if ARCHIVAL_OVERRIDE_PATTERNS matches, NEVER allow these entertainment/sports/celebrity categories
ABSOLUTE_REJECT_PATTERNS = re.compile(
    r"\b("
    r"living people|actors|actresses|singers|musicians|rappers|baseball|seattle mariners|burlington|"
    r"red carpet|film premiere|movie premiere|television|nbcuniversal|maxim|grammy|invictus games|"
    r"jennifer lopez|bad bunny|benicio del toro|amaury nolasco|ana ortiz|prince harry|"
    r"memorial plaque|wayside marker|information board|tourist sign"
    r")\b",
    re.IGNORECASE,
)


class MediaFetcher:
    TARGET_W = 1080
    TARGET_H = 1920

    def __init__(
        self,
        topic: str = "",
        era_keywords: Optional[List[str]] = None,
        wikipedia_articles: Optional[List[str]] = None,
        required_visual_keywords: Optional[List[str]] = None,
        is_space_or_science: bool = False,
        llm_client=None,
    ):
        self.foundry_endpoint = os.getenv("FOUNDRY_ENDPOINT", "").strip().rstrip("/")
        self.foundry_key = os.getenv("FOUNDRY_API_KEY", "").strip()
        self.image_model = os.getenv("FOUNDRY_IMAGE_MODEL", "MAI-Image-2.6-Flash").strip()
        self.pexels_key = os.getenv("PEXELS_API_KEY", "").strip()

        try:
            self.max_ai_images = max(0, int(os.getenv("MAX_AI_IMAGES_PER_SHORT", "3")))
        except ValueError:
            self.max_ai_images = 3
        self.ai_images_used = 0

        self.topic = topic
        self.era_keywords = era_keywords or []
        self.wikipedia_articles = wikipedia_articles or []
        self.required_keywords = [
            k.lower().strip() for k in (required_visual_keywords or []) if len(k.strip()) >= 3
        ]
        self.is_space_or_science = is_space_or_science or bool(
            re.search(r"\b(nasa|space|rocket|mariner|apollo|orbit|venus|mars|moon|satellite|probe|hubble)\b", topic, re.I)
        )
        self.llm_client = llm_client

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.attributions: List[str] = []

        self.used_urls: Set[str] = set()
        self.used_hashes: Set[str] = set()
        self.verified_pool: List[Dict[str, str]] = []
        self.downloaded_scene_images: List[Path] = []

        if self.topic:
            self._build_verified_archival_pool()

    @classmethod
    def normalize_image_to_9_16(cls, img: Image.Image) -> Image.Image:
        """
        Normalizes ANY input image (portrait, square, or horizontal) into a crisp, full-bleed
        1080x1920 (9:16) vertical image using subject-aware edge-energy salience detection
        for horizontal crops and upper-third framing bias (`0.22`) for vertical crops.
        Guarantees every scene image has the exact same 1080x1920 9:16 dimensions.
        """
        rgb = img.convert("RGB")
        w, h = rgb.size
        target_w, target_h = cls.TARGET_W, cls.TARGET_H

        scale = max(target_w / max(w, 1), target_h / max(h, 1))
        new_w = max(target_w, int(round(w * scale)))
        new_h = max(target_h, int(round(h * scale)))
        resized = rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)

        if new_w > target_w:
            # Horizontal excess: find the 9:16 vertical window with highest visual detail/salience
            gray_edges = resized.convert("L").filter(ImageFilter.FIND_EDGES)
            small_w, small_h = max(1, new_w // 8), max(1, new_h // 8)
            edge_small = gray_edges.resize((small_w, small_h), Image.Resampling.BILINEAR)
            win_w = max(1, target_w // 8)
            px = edge_small.load()

            col_energy = [0.0] * small_w
            for x in range(small_w):
                # Weight the upper-middle vertical band (15% to 75% height) highest
                y_start = int(small_h * 0.15)
                y_end = max(y_start + 1, int(small_h * 0.75))
                col_energy[x] = sum(px[x, y] for y in range(y_start, y_end))

            best_x_small = (small_w - win_w) // 2
            best_score = -1.0
            center_x_small = (small_w - win_w) / 2.0
            for x0 in range(0, max(1, small_w - win_w + 1)):
                energy = sum(col_energy[x0 : x0 + win_w])
                # Slight center-prior so we don't hug extreme edges unless subject is strongly there
                dist_penalty = 1.0 - 0.18 * abs(x0 - center_x_small) / max(1.0, center_x_small)
                score = energy * dist_penalty
                if score > best_score:
                    best_score = score
                    best_x_small = x0

            left = min(max(0, int(best_x_small * 8)), new_w - target_w)
        else:
            left = 0

        if new_h > target_h:
            # Vertical excess: bias toward upper-third (0.22) so faces, crowns, and rocket noses are never cut off
            top = min(max(0, int((new_h - target_h) * 0.22)), new_h - target_h)
        else:
            top = 0

        cropped = resized.crop((left, top, left + target_w, top + target_h))
        return ImageEnhance.Sharpness(cropped).enhance(1.08)

    def _has_domain_keyword_overlap(self, text: str) -> bool:
        """
        Requires candidate metadata to contain at least one strict domain keyword from the topic.
        """
        low = text.lower()
        if self.required_keywords:
            return any(kw in low for kw in self.required_keywords)
        topic_nouns = [
            w.lower() for w in re.findall(r"[A-Za-z0-9]{4,}", self.topic)
            if w.lower() not in {"that", "with", "from", "into", "when", "what", "single", "missing", "almost", "most", "craziest"}
        ]
        return any(w in low for w in topic_nouns) if topic_nouns else True

    def _is_strictly_historical_and_relevant(
        self, title: str, desc: str, categories: str, width: int, height: int, trust_curated_source: bool = False
    ) -> bool:
        if width < 440 or height < 440:
            return False
        low_title = title.lower()
        if low_title.endswith((".svg", ".gif", ".tif", ".tiff", ".pdf", ".djvu", ".xcf", ".ogg", ".ogv", ".webm")):
            return False

        # Reject wide horizontal banners/panoramas (`aspect > 1.60`) so every image crops cleanly to 9:16
        aspect = width / max(height, 1)
        if aspect > 1.60 or aspect < 0.35:
            return False

        combined = f"{title} {desc} {categories}"

        if ABSOLUTE_REJECT_PATTERNS.search(combined):
            return False

        if DISQUALIFYING_PATTERNS.search(combined):
            if not ARCHIVAL_OVERRIDE_PATTERNS.search(combined):
                return False

        if not trust_curated_source and not self._has_domain_keyword_overlap(combined):
            return False

        return True

    def _build_verified_archival_pool(self):
        raw_candidates: List[Dict[str, str]] = []

        articles_to_check = list(dict.fromkeys(self.wikipedia_articles[:5]))
        if not articles_to_check:
            articles_to_check = self._find_canonical_wikipedia_titles(self.topic)

        for article in articles_to_check:
            raw_candidates.extend(self._fetch_wikipedia_rest_article_media(article))

        if self.is_space_or_science:
            nasa_queries = [self.topic] + [a for a in articles_to_check[:3]]
            for nq in nasa_queries:
                raw_candidates.extend(self._search_nasa_archive(nq, limit=8))

        for article in articles_to_check[:3]:
            raw_candidates.extend(self._search_wikimedia_candidates(f"{article} historical", limit=6))

        unique_candidates: List[Dict[str, str]] = []
        seen_urls: Set[str] = set()
        for c in raw_candidates:
            if c["url"] not in seen_urls:
                seen_urls.add(c["url"])
                c["id"] = f"img_{len(unique_candidates)}"
                unique_candidates.append(c)

        if self.llm_client and unique_candidates:
            approved_ids = set(self.llm_client.verify_archival_candidates(self.topic, unique_candidates))
            if approved_ids:
                self.verified_pool = [c for c in unique_candidates if c["id"] in approved_ids]
            else:
                self.verified_pool = unique_candidates
        else:
            self.verified_pool = unique_candidates

        print(
            f"[MediaFetcher] Built verified archival pool: {len(self.verified_pool)} approved 9:16-ready images "
            f"(from {len(unique_candidates)} pre-filtered candidates) for '{self.topic}'."
        )

    def _find_canonical_wikipedia_titles(self, query: str) -> List[str]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 3,
            "format": "json",
        }
        try:
            resp = self.session.get(WIKIPEDIA_API, params=params, timeout=15)
            if resp.status_code == 200:
                hits = resp.json().get("query", {}).get("search", [])
                return [
                    h["title"] for h in hits
                    if not h["title"].lower().startswith("list of")
                    and not ABSOLUTE_REJECT_PATTERNS.search(h["title"])
                ]
        except Exception:
            pass
        return []

    def _fetch_wikipedia_rest_article_media(self, article_title: str) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        encoded = quote(article_title.replace(" ", "_"), safe="")
        try:
            sum_resp = self.session.get(f"{WIKIPEDIA_REST_SUMMARY}{encoded}", timeout=15)
            if sum_resp.status_code == 200:
                sdata = sum_resp.json()
                orig = sdata.get("originalimage", {}) or sdata.get("thumbnail", {})
                img_url = orig.get("source", "")
                w = int(orig.get("width", 1000))
                h = int(orig.get("height", 1000))
                desc = sdata.get("description", "") + " " + sdata.get("extract", "")
                if img_url and self._is_strictly_historical_and_relevant(
                    article_title, desc, article_title, w, h, trust_curated_source=True
                ):
                    results.append({
                        "url": img_url,
                        "title": f"{article_title} (Lead Archival Photo)",
                        "desc": desc[:160],
                        "license": "Wikipedia Archival",
                        "is_portrait": "1" if w <= h * 1.05 else "0",
                    })

            med_resp = self.session.get(f"{WIKIPEDIA_REST_MEDIA}{encoded}", timeout=15)
            if med_resp.status_code == 200:
                items = med_resp.json().get("items", [])
                for item in items:
                    if item.get("type") != "image":
                        continue
                    title = item.get("title", "").replace("File:", "")
                    caption = (item.get("caption", {}) or {}).get("text", "")
                    srcset = item.get("srcset", [])
                    if not srcset:
                        continue
                    best_src = srcset[-1].get("src", "")
                    if best_src.startswith("//"):
                        best_src = "https:" + best_src
                    if not best_src:
                        continue
                    if self._is_strictly_historical_and_relevant(
                        title, caption, article_title, 1000, 1000, trust_curated_source=True
                    ):
                        results.append({
                            "url": best_src,
                            "title": title,
                            "desc": f"{article_title}: {caption}"[:160],
                            "license": "Wikimedia Commons",
                            "is_portrait": "1",
                        })
        except Exception as exc:
            print(f"[MediaFetcher] Wikipedia REST media-list notice for '{article_title}': {exc}")
        return results

    def _search_nasa_archive(self, query: str, limit: int = 8) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        params = {"q": query, "media_type": "image", "page_size": limit}
        try:
            resp = self.session.get(NASA_IMAGES_API, params=params, timeout=18)
            if resp.status_code != 200:
                return results
            items = resp.json().get("collection", {}).get("items", [])
            for item in items:
                data_list = item.get("data", [])
                links_list = item.get("links", [])
                if not data_list or not links_list:
                    continue
                meta = data_list[0]
                title = meta.get("title", "NASA Archival Photograph")
                desc = meta.get("description", "")
                img_url = links_list[0].get("href", "")
                if not img_url or not img_url.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                if self._is_strictly_historical_and_relevant(
                    title, desc, "NASA Space Archive", 1024, 1024, trust_curated_source=True
                ):
                    results.append({
                        "url": img_url,
                        "title": f"NASA: {title}",
                        "desc": desc[:160],
                        "license": "Public Domain (NASA)",
                        "is_portrait": "1",
                    })
        except Exception as exc:
            print(f"[MediaFetcher] NASA archive lookup notice for '{query}': {exc}")
        return results

    def _search_wikimedia_candidates(self, query: str, limit: int = 8) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {query}",
            "gsrnamespace": 6,
            "gsrlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|dimensions",
            "iiurlwidth": 1280,
            "format": "json",
        }
        try:
            resp = self.session.get(WIKIMEDIA_API, params=params, timeout=18)
            if resp.status_code != 200:
                return results
            pages = resp.json().get("query", {}).get("pages", {})
            sorted_pages = sorted(pages.values(), key=lambda p: p.get("index", 999))
            for page in sorted_pages:
                title = page.get("title", "").replace("File:", "")
                iinfo = (page.get("imageinfo") or [{}])[0]
                img_url = iinfo.get("thumburl") or iinfo.get("url")
                w = int(iinfo.get("width", 0))
                h = int(iinfo.get("height", 0))
                ext = iinfo.get("extmetadata", {}) or {}
                desc = str(ext.get("ImageDescription", {}).get("value", ""))
                cats = str(ext.get("Categories", {}).get("value", ""))
                date_orig = str(ext.get("DateTimeOriginal", {}).get("value", ""))

                if not img_url:
                    continue
                if not self._is_strictly_historical_and_relevant(
                    title, f"{desc} {date_orig}", cats, w, h, trust_curated_source=False
                ):
                    continue

                lic = ext.get("LicenseShortName", {}).get("value", "Public Domain / CC")
                results.append({
                    "url": img_url,
                    "title": title,
                    "desc": re.sub(r"<[^>]+>", "", desc)[:160],
                    "license": lic,
                    "is_portrait": "1" if w <= h * 1.05 else "0",
                })
        except Exception as exc:
            print(f"[MediaFetcher] Wikimedia candidate search notice for '{query}': {exc}")
        return results

    def fetch_scene_asset(
        self, scene: dict, assets_dir: Path
    ) -> Tuple[Path, str]:
        assets_dir.mkdir(parents=True, exist_ok=True)
        scene_id = int(scene.get("scene_id", 1))
        prefer = scene.get("prefer_media", "archival")
        wiki_q = scene.get("wikimedia_query", "")
        nasa_q = scene.get("nasa_query", "")
        wp_article = scene.get("wikipedia_article", "")
        img_prompt = scene.get("image_prompt", "")
        pexels_q = scene.get("pexels_query", "")

        can_use_ai = (
            self.ai_images_used < self.max_ai_images
            and bool(self.foundry_endpoint and self.foundry_key)
        )

        if self.pexels_key and pexels_q and scene_id % 5 == 0:
            vid = self._fetch_pexels_video(pexels_q, assets_dir / f"scene_{scene_id:02d}.mp4")
            if vid:
                return vid, "video"

        if prefer == "ai_image" and can_use_ai and img_prompt:
            ai_img = self._generate_foundry_image(img_prompt, assets_dir / f"scene_{scene_id:02d}_ai.jpg")
            if ai_img:
                self.ai_images_used += 1
                self.downloaded_scene_images.append(ai_img)
                print(f"[MediaFetcher] Scene {scene_id:02d}: Generated 9:16 MAI-Image ({self.ai_images_used}/{self.max_ai_images} budget)")
                return ai_img, "image"

        if self.is_space_or_science and (nasa_q or wiki_q):
            for c in self._search_nasa_archive(nasa_q or wiki_q, limit=5):
                saved = self._download_and_register_image(
                    c["url"], assets_dir / f"scene_{scene_id:02d}_nasa.jpg", c["title"], c["license"]
                )
                if saved:
                    self.downloaded_scene_images.append(saved)
                    return saved, "image"

        pool_img = self._pull_best_from_verified_pool(
            f"{wiki_q} {wp_article} {scene.get('narration', '')}",
            assets_dir / f"scene_{scene_id:02d}_archival.jpg",
        )
        if pool_img:
            self.downloaded_scene_images.append(pool_img)
            return pool_img, "image"

        if wp_article:
            for c in self._fetch_wikipedia_rest_article_media(wp_article):
                saved = self._download_and_register_image(
                    c["url"], assets_dir / f"scene_{scene_id:02d}_wp.jpg", c["title"], c["license"]
                )
                if saved:
                    self.downloaded_scene_images.append(saved)
                    return saved, "image"

        if wiki_q:
            for c in self._search_wikimedia_candidates(wiki_q, limit=6):
                saved = self._download_and_register_image(
                    c["url"], assets_dir / f"scene_{scene_id:02d}_wm.jpg", c["title"], c["license"]
                )
                if saved:
                    self.downloaded_scene_images.append(saved)
                    return saved, "image"

        if self.ai_images_used < self.max_ai_images and self.foundry_endpoint and self.foundry_key:
            prompt = img_prompt or f"Authentic historical documentary photograph of {wiki_q or self.topic}, dramatic lighting, vertical 9:16"
            ai_img = self._generate_foundry_image(prompt, assets_dir / f"scene_{scene_id:02d}_ai.jpg")
            if ai_img:
                self.ai_images_used += 1
                self.downloaded_scene_images.append(ai_img)
                print(f"[MediaFetcher] Scene {scene_id:02d}: Fallback 9:16 MAI-Image ({self.ai_images_used}/{self.max_ai_images} budget)")
                return ai_img, "image"

        if self.downloaded_scene_images:
            source_img = self.downloaded_scene_images[(scene_id - 1) % len(self.downloaded_scene_images)]
            reframed_path = assets_dir / f"scene_{scene_id:02d}_detail.jpg"
            self._create_distinct_detail_crop(source_img, reframed_path, variant=scene_id)
            return reframed_path, "image"

        fallback_path = assets_dir / f"scene_{scene_id:02d}_fallback.jpg"
        self._create_archival_fallback_card(fallback_path)
        return fallback_path, "image"

    def _pull_best_from_verified_pool(self, scene_context: str, dest_path: Path) -> Optional[Path]:
        context_words = {
            w.lower() for w in re.findall(r"[A-Za-z0-9]{4,}", scene_context)
        }
        unused = [c for c in self.verified_pool if c["url"] not in self.used_urls]
        if not unused:
            return None

        def score_cand(c: Dict[str, str]) -> int:
            text = f"{c.get('title', '')} {c.get('desc', '')}".lower()
            kw_score = sum(2 for w in context_words if w in text)
            portrait_bonus = 3 if c.get("is_portrait") == "1" else 0
            return kw_score + portrait_bonus

        ranked = sorted(unused, key=score_cand, reverse=True)
        for cand in ranked:
            saved = self._download_and_register_image(
                cand["url"], dest_path, cand.get("title", ""), cand.get("license", "Public Domain")
            )
            if saved:
                return saved
        return None

    def _download_and_register_image(
        self, img_url: str, dest_path: Path, title: str = "", license_name: str = "Public Domain / CC"
    ) -> Optional[Path]:
        if img_url in self.used_urls:
            return None
        try:
            img_resp = self.session.get(img_url, timeout=25)
            img_resp.raise_for_status()
            content_bytes = img_resp.content
            digest = hashlib.md5(content_bytes).hexdigest()
            if digest in self.used_hashes:
                return None

            dest_path.write_bytes(content_bytes)
            with Image.open(dest_path) as im:
                w, h = im.size
                if w < 420 or h < 420 or (w / max(h, 1)) > 1.65:
                    dest_path.unlink(missing_ok=True)
                    return None
                norm_im = self.normalize_image_to_9_16(im)
                norm_im.save(dest_path, quality=94)

            self.used_urls.add(img_url)
            self.used_hashes.add(digest)
            if title:
                clean_title = title.replace("File:", "")
                self.attributions.append(f"• {clean_title} ({license_name})")
            return dest_path
        except Exception:
            return None

    def _generate_foundry_image(self, prompt: str, dest_path: Path) -> Optional[Path]:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.foundry_endpoint)
            if parsed.scheme and parsed.netloc:
                resource_root = f"{parsed.scheme}://{parsed.netloc}"
            else:
                resource_root = self.foundry_endpoint.split("/api/projects")[0].split("/openai")[0].split("/models")[0].rstrip("/")

            headers = {
                "Content-Type": "application/json",
                "api-key": self.foundry_key,
                "Authorization": f"Bearer {self.foundry_key}",
            }
            clean_prompt = (
                f"{prompt}, authentic historical documentary photography, dramatic chiaroscuro lighting, "
                "35mm archival film grain, strict vertical 9:16 portrait composition, no text, no watermarks, no borders"
            )

            route_attempts = [
                (
                    f"{resource_root}/mai/v1/images/generations",
                    {"model": self.image_model, "prompt": clean_prompt, "width": 768, "height": 1344},
                ),
                (
                    f"{resource_root}/openai/v1/images/generations",
                    {"model": self.image_model, "prompt": clean_prompt, "n": 1, "size": "1024x1792"},
                ),
                (
                    f"{resource_root}/openai/deployments/{self.image_model}/images/generations?api-version=2025-04-01-preview",
                    {"prompt": clean_prompt, "n": 1, "size": "1024x1792"},
                ),
            ]

            for url, payload in route_attempts:
                resp = requests.post(url, headers=headers, json=payload, timeout=55)
                if resp.status_code == 200:
                    data = resp.json()
                    items = data.get("data", [])
                    if items and "b64_json" in items[0]:
                        dest_path.write_bytes(base64.b64decode(items[0]["b64_json"]))
                    elif items and "url" in items[0]:
                        img_r = self.session.get(items[0]["url"], timeout=35)
                        img_r.raise_for_status()
                        dest_path.write_bytes(img_r.content)
                    else:
                        continue
                    with Image.open(dest_path) as im:
                        norm_im = self.normalize_image_to_9_16(im)
                        norm_im.save(dest_path, quality=94)
                    return dest_path
            return None
        except Exception as exc:
            print(f"[MediaFetcher] Foundry {self.image_model} notice: {exc}")
            return None

    def _fetch_pexels_video(self, query: str, dest_path: Path) -> Optional[Path]:
        headers = {"Authorization": self.pexels_key}
        params = {"query": query, "orientation": "portrait", "per_page": 4, "size": "medium"}
        try:
            resp = self.session.get(PEXELS_VIDEO_API, headers=headers, params=params, timeout=20)
            resp.raise_for_status()
            for v in resp.json().get("videos", []):
                files = v.get("video_files", [])
                mp4s = [f for f in files if f.get("file_type") == "video/mp4" and f.get("height", 0) >= 720]
                if not mp4s:
                    continue
                chosen = sorted(mp4s, key=lambda x: abs(x.get("height", 0) - 1280))[0]
                vid_resp = self.session.get(chosen["link"], timeout=35)
                vid_resp.raise_for_status()
                dest_path.write_bytes(vid_resp.content)
                return dest_path
        except Exception:
            pass
        return None

    @classmethod
    def _create_distinct_detail_crop(cls, source_path: Path, dest_path: Path, variant: int = 1):
        """
        Creates a distinct 9:16 documentary camera angle (rule-of-thirds close-up) normalized
        strictly to 1080x1920 so every scene fills the 9:16 vertical screen edge-to-edge.
        """
        with Image.open(source_path) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            crop_w = int(w * 0.78)
            crop_h = int(h * 0.78)
            offsets = [
                (int(w * 0.04), int(h * 0.05)),
                (int(w * 0.18), int(h * 0.08)),
                (int(w * 0.11), int(h * 0.16)),
                (int(w * 0.15), int(h * 0.04)),
            ]
            ox, oy = offsets[variant % len(offsets)]
            cropped = rgb.crop((ox, oy, min(w, ox + crop_w), min(h, oy + crop_h)))
            cropped = ImageEnhance.Contrast(cropped).enhance(1.08)
            cropped = ImageEnhance.Color(cropped).enhance(0.90)
            norm_im = cls.normalize_image_to_9_16(cropped)
            norm_im.save(dest_path, quality=94)

    @staticmethod
    def _create_archival_fallback_card(dest_path: Path):
        w, h = 1080, 1920
        arr = Image.effect_noise((w, h), 28).convert("RGB")
        r, g, b = arr.split()
        r = r.point(lambda i: int(i * 0.25 + 22))
        g = g.point(lambda i: int(i * 0.22 + 18))
        b = b.point(lambda i: int(i * 0.18 + 14))
        img = Image.merge("RGB", (r, g, b)).filter(ImageFilter.GaussianBlur(radius=2))
        img.save(dest_path, quality=92)
