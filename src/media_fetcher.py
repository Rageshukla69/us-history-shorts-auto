"""
Multi-Source Visual Fetcher tailored to your exact Heroku Config Vars:
1. Wikipedia / Wikimedia Commons API: FREE & PUBLIC — requires ZERO API keys or secrets!
   Fetches real historical photographs, battle paintings, maps, and engravings.
2. Microsoft Foundry `MAI-Image-2.6-Flash` (`FOUNDRY_ENDPOINT` + `FOUNDRY_API_KEY` + `FOUNDRY_IMAGE_MODEL`):
   Uses the exact Foundry Image model already saved in your Heroku config to generate bespoke 9:16 vertical
   historical visuals when `prefer_media == "ai_image"` or when Wikimedia has no photo for an obscure scene!
3. Optional Pexels Video API (`PEXELS_API_KEY`): Only used if you optionally paste a free Pexels key;
   completely skipped if left blank.
"""
import base64
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple
import requests
from PIL import Image, ImageFilter

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
PEXELS_VIDEO_API = "https://api.pexels.com/videos/search"
USER_AGENT = "USHistoryShortsBot/2.0 (https://github.com/education/students)"


class MediaFetcher:
    def __init__(self):
        self.foundry_endpoint = os.getenv("FOUNDRY_ENDPOINT", "").strip().rstrip("/")
        self.foundry_key = os.getenv("FOUNDRY_API_KEY", "").strip()
        self.image_model = os.getenv("FOUNDRY_IMAGE_MODEL", "MAI-Image-2.6-Flash").strip()
        self.pexels_key = os.getenv("PEXELS_API_KEY", "").strip()

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.attributions: List[str] = []

    def fetch_scene_asset(
        self, scene: dict, assets_dir: Path
    ) -> Tuple[Path, str]:
        assets_dir.mkdir(parents=True, exist_ok=True)
        scene_id = scene.get("scene_id", 1)
        prefer = scene.get("prefer_media", "wikimedia")
        wiki_q = scene.get("wikimedia_query", "")
        img_prompt = scene.get("image_prompt", "")
        pexels_q = scene.get("pexels_query", "")

        # 1. Optional Pexels motion video if PEXELS_API_KEY is provided and scene requests video
        if self.pexels_key and pexels_q and scene_id % 4 == 0:
            video_path = self._fetch_pexels_video(pexels_q, assets_dir / f"scene_{scene_id:02d}.mp4")
            if video_path:
                return video_path, "video"

        # 2. If scene prefers MAI-Image-2.6-Flash AI visual, generate via Foundry Image endpoint first
        if prefer == "ai_image" and self.foundry_endpoint and self.foundry_key and img_prompt:
            ai_img = self._generate_foundry_image(img_prompt, assets_dir / f"scene_{scene_id:02d}_ai.jpg")
            if ai_img:
                return ai_img, "image"

        # 3. Query Wikipedia & Wikimedia Commons (No API key needed!)
        if wiki_q:
            img_path = self._fetch_wikimedia_image(wiki_q, assets_dir / f"scene_{scene_id:02d}.jpg")
            if img_path:
                return img_path, "image"

            # Try Wikipedia page lead image search
            wp_img = self._fetch_wikipedia_page_image(wiki_q, assets_dir / f"scene_{scene_id:02d}_wp.jpg")
            if wp_img:
                return wp_img, "image"

        # 4. Fallback to Microsoft Foundry `MAI-Image-2.6-Flash` if Wikimedia had no match
        if self.foundry_endpoint and self.foundry_key:
            prompt = img_prompt or f"Archival historical documentary photograph of {wiki_q or scene.get('narration', '')}, dramatic lighting, vertical 9:16"
            ai_img = self._generate_foundry_image(prompt, assets_dir / f"scene_{scene_id:02d}_ai.jpg")
            if ai_img:
                return ai_img, "image"

        # 5. Try broader 2-word Wikimedia search
        short_q = " ".join(re.findall(r"[A-Za-z0-9]+", wiki_q)[:2])
        if short_q:
            img_path = self._fetch_wikimedia_image(short_q, assets_dir / f"scene_{scene_id:02d}.jpg")
            if img_path:
                return img_path, "image"

        # 6. Guaranteed archival slate fallback so render never crashes
        fallback_path = assets_dir / f"scene_{scene_id:02d}_fallback.jpg"
        self._create_archival_fallback_card(fallback_path)
        return fallback_path, "image"

    def _generate_foundry_image(self, prompt: str, dest_path: Path) -> Optional[Path]:
        """
        Calls Microsoft Foundry `MAI-Image-2.6-Flash` using `FOUNDRY_ENDPOINT` + `FOUNDRY_API_KEY`.
        """
        try:
            url = self.foundry_endpoint
            if "/images/generations" not in url:
                if "openai.azure.com" in url and "/deployments/" not in url:
                    url = f"{url}/openai/deployments/{self.image_model}/images/generations?api-version=2024-05-01-preview"
                else:
                    url = f"{url}/images/generations"
                    if "services.ai.azure.com" in url and "api-version=" not in url:
                        url += "?api-version=2024-05-01-preview"

            headers = {
                "Content-Type": "application/json",
                "api-key": self.foundry_key,
                "Authorization": f"Bearer {self.foundry_key}",
            }
            payload = {
                "model": self.image_model,
                "prompt": f"{prompt}. Clean historical composition, no text, no watermarks, no labels.",
                "n": 1,
                "size": "1024x1792",
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=45)
            if resp.status_code >= 400:
                # Retry with standard 1024x1024 if endpoint restricts aspect ratio string
                payload["size"] = "1024x1024"
                resp = requests.post(url, headers=headers, json=payload, timeout=45)
            resp.raise_for_status()

            data = resp.json().get("data", [{}])[0]
            if "b64_json" in data:
                dest_path.write_bytes(base64.b64decode(data["b64_json"]))
            elif "url" in data:
                img_r = self.session.get(data["url"], timeout=30)
                img_r.raise_for_status()
                dest_path.write_bytes(img_r.content)
            else:
                return None

            with Image.open(dest_path) as im:
                im.convert("RGB").save(dest_path, quality=92)
            return dest_path
        except Exception as exc:
            print(f"[MediaFetcher] Foundry {self.image_model} notice: {exc}")
            return None

    def _fetch_wikimedia_image(self, query: str, dest_path: Path) -> Optional[Path]:
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {query}",
            "gsrnamespace": 6,
            "gsrlimit": 5,
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
                iinfo = page.get("imageinfo", [{}])[0]
                img_url = iinfo.get("thumburl") or iinfo.get("url")
                width = iinfo.get("width", 0)
                height = iinfo.get("height", 0)
                if not img_url or width < 450 or height < 450:
                    continue
                if img_url.lower().endswith((".svg", ".gif", ".tif", ".pdf")):
                    continue

                img_resp = self.session.get(img_url, timeout=25)
                img_resp.raise_for_status()
                dest_path.write_bytes(img_resp.content)

                with Image.open(dest_path) as im:
                    im.convert("RGB").save(dest_path, quality=92)

                title = page.get("title", "").replace("File:", "")
                ext = iinfo.get("extmetadata", {})
                license_name = ext.get("LicenseShortName", {}).get("value", "Public Domain / CC")
                self.attributions.append(f"• {title} ({license_name}) via Wikimedia Commons")
                return dest_path
        except Exception as exc:
            print(f"[MediaFetcher] Wikimedia lookup notice for '{query}': {exc}")
        return None

    def _fetch_wikipedia_page_image(self, query: str, dest_path: Path) -> Optional[Path]:
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": 3,
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
                img_url = page.get("original", {}).get("source") or page.get("thumbnail", {}).get("source")
                if not img_url or img_url.lower().endswith((".svg", ".gif", ".tif", ".pdf")):
                    continue
                img_resp = self.session.get(img_url, timeout=20)
                img_resp.raise_for_status()
                dest_path.write_bytes(img_resp.content)
                with Image.open(dest_path) as im:
                    im.convert("RGB").save(dest_path, quality=92)
                return dest_path
        except Exception:
            pass
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
