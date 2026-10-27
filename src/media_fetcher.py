"""
Multi-Source Archival & Credit-Saving Visual Fetcher tailored to your exact Heroku Config Vars:
1. Credit Conservation Guardrail (`MAX_AI_IMAGES_PER_SHORT`, default `2`):
   Caps `MAI-Image-2.6-Flash` calls so your Microsoft Foundry credits last 8x-10x longer!
   Can even be set to `0` or `1` in GitHub Secrets / `.env`.
2. Topic-Anchored Relevance Filtering & Modern-Celebrity Blacklist:
   Prevents ambiguous words (like "rubber", "actors", "illusion", "looks real") from pulling
   random modern celebrities (e.g., Prince Harry at Invictus Games), modern tourist plaques,
   or unrelated stock photos.
3. Deep Public Archival Pool & Zero-Duplicate Scene Cropping:
   Pre-fetches the full gallery of historical photographs from the topic's main Wikipedia articles
   and Wikimedia Commons categories, deduplicates every downloaded URL across all 16-18 scenes,
   and creates distinct cinematic detail crops when needed without spending AI credits.
"""
import base64
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import requests
from PIL import Image, ImageEnhance, ImageFilter

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
PEXELS_VIDEO_API = "https://api.pexels.com/videos/search"
USER_AGENT = "USHistoryShortsBot/3.0 (https://github.com/education/students)"

IRRELEVANT_TITLE_PATTERNS = re.compile(
    r"\b("
    r"prince harry|meghan markle|invictus games|red carpet|paparazzi|selfie|"
    r"press conference|podcast|youtuber|influencer|tiktok|instagram|"
    r"memorial plaque|information board|tourist sign|wayside marker|historical marker|"
    r"commemorative plaque|road sign|street sign|parking lot|shopping mall|"
    r"logo|icon|coat of arms svg|flag of|map locator|blank map|"
    r"user:|talk:|template:|file icon|wikimedia|creative commons"
    r")\b",
    re.IGNORECASE,
)

GENERIC_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "were", "was", "are",
    "its", "their", "they", "into", "until", "about", "near", "even", "made",
    "make", "built", "look", "looked", "looks", "real", "fake", "fakes",
    "rubber", "secret", "strangest", "mission", "distance", "remarkably",
    "illusion", "illusions", "actors", "men", "unit", "units", "history",
    "vanished", "afterward", "classified", "declassified", "appeared", "nowhere",
}


class MediaFetcher:
    def __init__(self, topic: str = "", era_keywords: Optional[List[str]] = None):
        self.foundry_endpoint = os.getenv("FOUNDRY_ENDPOINT", "").strip().rstrip("/")
        self.foundry_key = os.getenv("FOUNDRY_API_KEY", "").strip()
        self.image_model = os.getenv("FOUNDRY_IMAGE_MODEL", "MAI-Image-2.6-Flash").strip()
        self.pexels_key = os.getenv("PEXELS_API_KEY", "").strip()

        try:
            self.max_ai_images = max(0, int(os.getenv("MAX_AI_IMAGES_PER_SHORT", "2")))
        except ValueError:
            self.max_ai_images = 2
        self.ai_images_used = 0

        self.topic = topic
        self.era_keywords = era_keywords or []
        self.topic_anchor = self._extract_topic_anchor(topic, self.era_keywords)

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.attributions: List[str] = []

        self.used_urls: Set[str] = set()
        self.used_hashes: Set[str] = set()
        self.topic_image_pool: List[Dict[str, str]] = []
        self.downloaded_scene_images: List[Path] = []

        if self.topic:
            self._warm_up_topic_archival_pool()

    @staticmethod
    def _extract_topic_anchor(topic: str, era_keywords: List[str]) -> str:
        """
        Extracts the core historical entity and year/era from the topic title so every search query
        remains anchored to the historical subject (e.g. 'Ghost Army World War II').
        """
        if not topic:
            return " ".join(era_keywords[:3]) if era_keywords else "United States history archival"
        cleaned = re.sub(
            r"\b(The|How|Why|When|Secret|Craziest|Most|Bizarre|Mystery|of|in|That|Fooled|Almost|Single|Entire|With)\b",
            " ",
            topic,
            flags=re.IGNORECASE,
        )
        words = [w for w in re.findall(r"[A-Za-z0-9]+", cleaned) if len(w) >= 2]
        anchor = " ".join(words[:5])
        return anchor or topic[:45]

    def _sanitize_and_anchor_query(self, raw_query: str, scene: dict) -> str:
        """
        Cleans a scene's search query, strips generic metaphor words (like 'rubber' or 'looks real'),
        and appends the topic anchor if the query lacks historical specificity.
        """
        tokens = [
            w for w in re.findall(r"[A-Za-z0-9]+", raw_query)
            if w.lower() not in GENERIC_STOPWORDS and len(w) > 1
        ]
        if not tokens:
            return self.topic_anchor

        cleaned_q = " ".join(tokens)

        anchor_words = {
            w.lower() for w in re.findall(r"[A-Za-z0-9]+", f"{self.topic_anchor} {' '.join(self.era_keywords)}")
            if len(w) > 2
        }
        query_words = {w.lower() for w in tokens}

        if anchor_words and not (query_words & anchor_words):
            short_anchor = " ".join(self.topic_anchor.split()[:3])
            cleaned_q = f"{short_anchor} {cleaned_q}".strip()

        return cleaned_q

    def _is_candidate_valid_and_relevant(self, title: str, extmeta: dict, width: int, height: int) -> bool:
        if width < 480 or height < 420:
            return False
        low_title = title.lower()
        if low_title.endswith((".svg", ".gif", ".tif", ".tiff", ".pdf", ".djvu", ".xcf", ".ogg", ".ogv", ".webm")):
            return False

        desc = ""
        categories = ""
        if isinstance(extmeta, dict):
            desc = str(extmeta.get("ImageDescription", {}).get("value", ""))
            categories = str(extmeta.get("Categories", {}).get("value", ""))

        combined_text = f"{title} {desc} {categories}"
        if IRRELEVANT_TITLE_PATTERNS.search(combined_text):
            return False

        aspect = width / max(height, 1)
        if aspect > 2.6 or aspect < 0.32:
            return False

        return True

    def _warm_up_topic_archival_pool(self):
        search_terms = [self.topic, self.topic_anchor]
        for term in search_terms:
            if not term:
                continue
            try:
                params = {
                    "action": "query",
                    "generator": "search",
                    "gsrsearch": term,
                    "gsrlimit": 3,
                    "prop": "images",
                    "imlimit": 30,
                    "format": "json",
                }
                resp = self.session.get(WIKIPEDIA_API, params=params, timeout=15)
                if resp.status_code != 200:
                    continue
                pages = resp.json().get("query", {}).get("pages", {})
                file_titles = []
                for _, page in pages.items():
                    for im in page.get("images", []):
                        t = im.get("title", "")
                        if t and not IRRELEVANT_TITLE_PATTERNS.search(t):
                            file_titles.append(t)

                if file_titles:
                    self._resolve_commons_files_to_pool(file_titles[:25])
            except Exception as exc:
                print(f"[MediaFetcher] Pool warm-up notice: {exc}")

        self._populate_pool_from_commons_search(f"{self.topic_anchor} historical photograph", limit=18)
        print(f"[MediaFetcher] Warmed up archival pool with {len(self.topic_image_pool)} verified historical images for '{self.topic_anchor}'.")

    def _resolve_commons_files_to_pool(self, titles: List[str]):
        if not titles:
            return
        params = {
            "action": "query",
            "titles": "|".join(titles[:25]),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|dimensions",
            "iiurlwidth": 1280,
            "format": "json",
        }
        try:
            resp = self.session.get(WIKIMEDIA_API, params=params, timeout=18)
            if resp.status_code != 200:
                return
            pages = resp.json().get("query", {}).get("pages", {})
            for _, page in pages.items():
                title = page.get("title", "")
                iinfo = (page.get("imageinfo") or [{}])[0]
                img_url = iinfo.get("thumburl") or iinfo.get("url")
                w = iinfo.get("width", 0)
                h = iinfo.get("height", 0)
                ext = iinfo.get("extmetadata", {})
                if img_url and self._is_candidate_valid_and_relevant(title, ext, w, h):
                    if not any(item["url"] == img_url for item in self.topic_image_pool):
                        lic = ext.get("LicenseShortName", {}).get("value", "Public Domain / CC")
                        self.topic_image_pool.append({
                            "url": img_url,
                            "title": title.replace("File:", ""),
                            "license": lic,
                        })
        except Exception:
            pass

    def _populate_pool_from_commons_search(self, query: str, limit: int = 12):
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
                return
            pages = resp.json().get("query", {}).get("pages", {})
            for _, page in pages.items():
                title = page.get("title", "")
                iinfo = (page.get("imageinfo") or [{}])[0]
                img_url = iinfo.get("thumburl") or iinfo.get("url")
                w = iinfo.get("width", 0)
                h = iinfo.get("height", 0)
                ext = iinfo.get("extmetadata", {})
                if img_url and self._is_candidate_valid_and_relevant(title, ext, w, h):
                    if not any(item["url"] == img_url for item in self.topic_image_pool):
                        lic = ext.get("LicenseShortName", {}).get("value", "Public Domain / CC")
                        self.topic_image_pool.append({
                            "url": img_url,
                            "title": title.replace("File:", ""),
                            "license": lic,
                        })
        except Exception:
            pass

    def fetch_scene_asset(
        self, scene: dict, assets_dir: Path
    ) -> Tuple[Path, str]:
        assets_dir.mkdir(parents=True, exist_ok=True)
        scene_id = int(scene.get("scene_id", 1))
        prefer = scene.get("prefer_media", "wikimedia")
        raw_wiki_q = scene.get("wikimedia_query", "")
        wp_article = scene.get("wikipedia_article", "")
        img_prompt = scene.get("image_prompt", "")

        anchored_q = self._sanitize_and_anchor_query(raw_wiki_q, scene)
        can_use_ai = (
            self.ai_images_used < self.max_ai_images
            and bool(self.foundry_endpoint and self.foundry_key)
        )

        if prefer == "ai_image" and can_use_ai and img_prompt:
            ai_img = self._generate_foundry_image(img_prompt, assets_dir / f"scene_{scene_id:02d}_ai.jpg")
            if ai_img:
                self.ai_images_used += 1
                self.downloaded_scene_images.append(ai_img)
                print(f"[MediaFetcher] Scene {scene_id:02d}: Used MAI-Image ({self.ai_images_used}/{self.max_ai_images} budget)")
                return ai_img, "image"

        if anchored_q:
            img_path = self._fetch_wikimedia_image(anchored_q, assets_dir / f"scene_{scene_id:02d}.jpg")
            if img_path:
                self.downloaded_scene_images.append(img_path)
                return img_path, "image"

        for wp_q in filter(None, [wp_article, anchored_q, self.topic_anchor]):
            wp_img = self._fetch_wikipedia_page_image(wp_q, assets_dir / f"scene_{scene_id:02d}_wp.jpg")
            if wp_img:
                self.downloaded_scene_images.append(wp_img)
                return wp_img, "image"

        pool_img = self._pull_from_topic_pool(assets_dir / f"scene_{scene_id:02d}_pool.jpg")
        if pool_img:
            self.downloaded_scene_images.append(pool_img)
            return pool_img, "image"

        if self.ai_images_used < self.max_ai_images and self.foundry_endpoint and self.foundry_key:
            prompt = img_prompt or f"Archival historical documentary photograph of {anchored_q}, {self.topic}, dramatic lighting, vertical 9:16"
            ai_img = self._generate_foundry_image(prompt, assets_dir / f"scene_{scene_id:02d}_ai.jpg")
            if ai_img:
                self.ai_images_used += 1
                self.downloaded_scene_images.append(ai_img)
                print(f"[MediaFetcher] Scene {scene_id:02d}: Fallback MAI-Image ({self.ai_images_used}/{self.max_ai_images} budget)")
                return ai_img, "image"

        if self.downloaded_scene_images:
            source_img = self.downloaded_scene_images[(scene_id - 1) % len(self.downloaded_scene_images)]
            reframed_path = assets_dir / f"scene_{scene_id:02d}_detail.jpg"
            self._create_distinct_detail_crop(source_img, reframed_path, variant=scene_id)
            return reframed_path, "image"

        fallback_path = assets_dir / f"scene_{scene_id:02d}_fallback.jpg"
        self._create_archival_fallback_card(fallback_path)
        return fallback_path, "image"

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
                if w < 420 or h < 380:
                    dest_path.unlink(missing_ok=True)
                    return None
                im.convert("RGB").save(dest_path, quality=92)

            self.used_urls.add(img_url)
            self.used_hashes.add(digest)
            if title:
                clean_title = title.replace("File:", "")
                self.attributions.append(f"• {clean_title} ({license_name}) via Wikimedia Commons")
            return dest_path
        except Exception:
            return None

    def _pull_from_topic_pool(self, dest_path: Path) -> Optional[Path]:
        for item in self.topic_image_pool:
            url = item["url"]
            if url in self.used_urls:
                continue
            saved = self._download_and_register_image(url, dest_path, item.get("title", ""), item.get("license", "Public Domain"))
            if saved:
                return saved
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
                "35mm archival film grain, vertical 9:16 composition, no text, no watermarks, no modern elements"
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
                        im.convert("RGB").save(dest_path, quality=92)
                    return dest_path
            return None
        except Exception as exc:
            print(f"[MediaFetcher] Foundry {self.image_model} notice: {exc}")
            return None

    def _fetch_wikimedia_image(self, query: str, dest_path: Path) -> Optional[Path]:
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {query}",
            "gsrnamespace": 6,
            "gsrlimit": 10,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|dimensions",
            "iiurlwidth": 1280,
            "format": "json",
        }
        try:
            resp = self.session.get(WIKIMEDIA_API, params=params, timeout=20)
            resp.raise_for_status()
            pages = resp.json().get("query", {}).get("pages", {})
            for _, page in pages.items():
                title = page.get("title", "")
                iinfo = (page.get("imageinfo") or [{}])[0]
                img_url = iinfo.get("thumburl") or iinfo.get("url")
                width = iinfo.get("width", 0)
                height = iinfo.get("height", 0)
                ext = iinfo.get("extmetadata", {})

                if not img_url or not self._is_candidate_valid_and_relevant(title, ext, width, height):
                    continue

                license_name = ext.get("LicenseShortName", {}).get("value", "Public Domain / CC")
                saved = self._download_and_register_image(img_url, dest_path, title, license_name)
                if saved:
                    return saved
        except Exception as exc:
            print(f"[MediaFetcher] Wikimedia lookup notice for '{query}': {exc}")
        return None

    def _fetch_wikipedia_page_image(self, query: str, dest_path: Path) -> Optional[Path]:
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": 5,
            "prop": "pageimages",
            "piprop": "original|thumbnail",
            "pithumbsize": 1280,
            "format": "json",
        }
        try:
            resp = self.session.get(WIKIPEDIA_API, params=params, timeout=15)
            resp.raise_for_status()
            pages = resp.json().get("query", {}).get("pages", {})
            for _, page in pages.items():
                page_title = page.get("title", "")
                if IRRELEVANT_TITLE_PATTERNS.search(page_title):
                    continue
                img_url = page.get("original", {}).get("source") or page.get("thumbnail", {}).get("source")
                if not img_url or IRRELEVANT_TITLE_PATTERNS.search(img_url):
                    continue
                if img_url.lower().endswith((".svg", ".gif", ".tif", ".pdf")):
                    continue
                saved = self._download_and_register_image(img_url, dest_path, page_title, "Wikipedia Archival")
                if saved:
                    return saved
        except Exception:
            pass
        return None

    @staticmethod
    def _create_distinct_detail_crop(source_path: Path, dest_path: Path, variant: int = 1):
        with Image.open(source_path) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            crop_w = int(w * 0.74)
            crop_h = int(h * 0.74)
            offsets = [
                (int(w * 0.05), int(h * 0.08)),
                (int(w * 0.20), int(h * 0.12)),
                (int(w * 0.12), int(h * 0.22)),
                (int(w * 0.18), int(h * 0.05)),
            ]
            ox, oy = offsets[variant % len(offsets)]
            cropped = rgb.crop((ox, oy, min(w, ox + crop_w), min(h, oy + crop_h)))
            cropped = ImageEnhance.Contrast(cropped).enhance(1.12)
            cropped = ImageEnhance.Color(cropped).enhance(0.82)
            cropped.save(dest_path, quality=92)

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
