"""
Multi-Archive & Up-to-10 MAI-Image-2.6-Flash Visual Fetcher (`src/media_fetcher.py` — v10.0):
1. Supports up to 10 bespoke 9:16 vertical `MAI-Image-2.6-Flash` cinema reconstructions per Short (`MAX_AI_IMAGES_PER_SHORT = 10` default).
2. Perceptual 64-bit Difference Hashing (`dHash`) blocks near-duplicate crops or resizes of the same Wikimedia/Wikipedia archival photo
   (eliminating the bug where the same artifact photo appeared 3 times in one Short!).
3. Strict Diagram / Schematic / Chart Rejection (`_is_valid_photographic_image`) rejects flat white/neon technical diagrams,
   charts, and low-entropy schematics so only true photographs or 9:16 AI cinema reconstructions ever reach the screen.
"""
import base64
import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
import requests

try:
    from archive_media import ArchiveMediaFinder, rank_candidates
except ImportError:
    from src.archive_media import ArchiveMediaFinder, rank_candidates

USER_AGENT = (
    "USHistoryShortsBot/10.0 "
    "(https://github.com/Rageshukla69/us-history-shorts-auto; DocumentaryPipeline/10.0)"
)

ABSOLUTE_REJECT_PATTERNS = re.compile(
    r"(shutterstock|getty[ _-]?images?|alamy[ _-]?stock|stock[ _-]?photo|watermark|"
    r"minecraft|fortnite|roblox|videogame|gameplay|anime|manga|hentai|cartoon|meme|parody|"
    r"lego|minifigure|action[ _-]?figure|toy[ _-]?review|cosplay|tiktok[ _-]?trend|"
    r"commons[ _-]?logo|wikimedia[ _-]?logo|qr[ _-]?code|barcode|"
    r"schematic|diagram|flowchart|blueprint[ _-]?diagram|cross[ _-]?section[ _-]?scheme|"
    r"vector[ _-]?diagram|chart|graph|plot|infographic)",
    re.IGNORECASE,
)

DISQUALIFYING_PATTERNS = re.compile(
    r"(\.svg|\.gif$|\.pdf|\.djvu|\.stl|\.webp$|"
    r"locator[ _-]map|location[ _-]map|"
    r"user[ _-]talk|wikimedia[ _-]commons[ _-]logo|commons-logo|wikisource|wiktionary|"
    r"book[ _-]cover|scan[ _-]of|text[ _-]of|transcript)",
    re.IGNORECASE,
)


class MediaFetcher:
    def __init__(
        self,
        topic: str,
        era_keywords: Optional[List[str]] = None,
        wikipedia_articles: Optional[List[str]] = None,
        video_pool_queries: Optional[List[str]] = None,
        required_visual_keywords: Optional[List[str]] = None,
        is_space_or_science: bool = False,
        llm_client: Any = None,
    ):
        self.topic = topic
        self.era_keywords = era_keywords or []
        self.wikipedia_articles = (wikipedia_articles or [])[:5]
        self.video_pool_queries = video_pool_queries or []
        self.is_space_or_science = is_space_or_science
        self.llm_client = llm_client

        raw_req = required_visual_keywords or []
        topic_words = [
            w.lower()
            for w in re.findall(
                r"[A-Za-z0-9]{4,}",
                f"{topic} {' '.join(self.wikipedia_articles)} {' '.join(self.video_pool_queries)}",
            )
            if w.lower() not in {"when", "that", "this", "with", "from", "into", "over", "were", "have", "history", "story", "most", "first"}
        ]
        self.domain_keywords: Set[str] = {w.lower() for w in raw_req if len(w) >= 3} | set(topic_words)

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.media_cache_dir = Path(os.getenv("MEDIA_CACHE_DIR", ".cache/media"))
        self.media_cache_dir.mkdir(parents=True, exist_ok=True)

        self.foundry_endpoint = (
            os.getenv("FOUNDRY_ENDPOINT")
            or os.getenv("FOUNDRY_TEXT_ENDPOINT")
            or ""
        ).strip().rstrip("/")
        self.foundry_key = (
            os.getenv("FOUNDRY_API_KEY")
            or os.getenv("FOUNDRY_TEXT_API_KEY")
            or ""
        ).strip()
        self.image_model = (
            os.getenv("FOUNDRY_IMAGE_MODEL") or "MAI-Image-2.6-Flash"
        ).strip()
        self.pexels_key = (os.getenv("PEXELS_API_KEY") or "").strip()

        raw_max_ai = os.getenv("MAX_AI_IMAGES_PER_SHORT")
        if raw_max_ai is None or str(raw_max_ai).strip() == "":
            raw_max_ai = os.getenv("MAX_AI_IMAGES_PER_VIDEO", "10")
        try:
            self.max_ai_images = max(0, int(str(raw_max_ai).strip()))
        except (TypeError, ValueError):
            self.max_ai_images = 10

        self.ai_images_used = 0
        self.video_clips_used = 0
        self.archival_photos_used = 0
        self.reframed_cuts_used = 0
        self.used_urls: Set[str] = set()
        self.used_hashes: Set[str] = set()
        self.used_dhashes: List[int] = []
        self.used_title_stems: Set[str] = set()
        self.attributions: List[str] = []
        self.topic_article_images: List[Dict[str, Any]] = []
        self.ai_generated_images: List[Path] = []
        self.downloaded_scene_images: List[Path] = []
        self.scene_audit_log: List[Dict[str, Any]] = []
        self.archive_finder = ArchiveMediaFinder(session=self.session, cache_dir=self.media_cache_dir)

        self._preload_topic_wikipedia_images()

    def _preload_topic_wikipedia_images(self):
        seen: Set[str] = set()
        articles = list(self.wikipedia_articles)
        if self.topic and self.topic not in articles:
            articles.append(self.topic)
        for art in articles[:5]:
            for item in self.archive_finder.fetch_wikipedia_article_images(art):
                u = item.get("url", "")
                t = item.get("title", "")
                if (
                    u
                    and u not in seen
                    and not ABSOLUTE_REJECT_PATTERNS.search(f"{t} {u}")
                    and not DISQUALIFYING_PATTERNS.search(u)
                ):
                    seen.add(u)
                    self.topic_article_images.append(item)
        print(f"[MediaFetcher] Preloaded {len(self.topic_article_images)} archival images from Wikipedia articles: {self.wikipedia_articles}")

    @staticmethod
    def _compute_dhash(rgb_img: Image.Image) -> int:
        """Computes a 64-bit perceptual difference hash (9x8 grayscale) to catch resized/cropped duplicate photos."""
        small = rgb_img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        arr = np.asarray(small, dtype=np.int16)
        diff = arr[:, 1:] > arr[:, :-1]
        bits = 0
        for b in diff.flatten():
            bits = (bits << 1) | int(bool(b))
        return bits

    def _is_perceptual_duplicate(self, dhash: int, max_hamming: int = 14) -> bool:
        for prev in self.used_dhashes:
            if bin(dhash ^ prev).count("1") <= max_hamming:
                return True
        return False

    @staticmethod
    def _normalize_title_stem(title: str) -> str:
        clean = re.sub(r"\.(jpg|jpeg|png|tif|tiff)$", "", title.lower().replace("file:", ""))
        clean = re.sub(r"\b(cropped|detail|fragment|view|front|back|side|closeup|close-up|01|02|1|2|3)\b", "", clean)
        tokens = [w for w in re.findall(r"[a-z]{4,}", clean)]
        return " ".join(tokens[:4])

    @staticmethod
    def _is_valid_photographic_image(rgb_img: Image.Image) -> bool:
        """
        Rejects blank/dark images AND flat computer diagrams/schematics (like white/gray background
        technical drawings with neon lines).
        """
        small = rgb_img.resize((200, 200), Image.Resampling.BILINEAR)
        arr_rgb = np.asarray(small, dtype=np.float32)
        gray = np.asarray(small.convert("L"), dtype=np.float32)
        mean_l = float(gray.mean())
        std_l = float(gray.std())
        if mean_l < 12.0 or std_l < 9.0:
            return False

        # Reject images with >48% near-white pixels (typical of diagrams, document scans, and schematics)
        white_ratio = float(np.mean(gray > 232.0))
        if white_ratio > 0.48:
            return False

        # Reject flat-shaded diagrams/charts where 4 quantized color buckets cover >68% of the canvas
        quant = (arr_rgb // 32).astype(np.int32)
        codes = quant[:, :, 0] * 64 + quant[:, :, 1] * 8 + quant[:, :, 2]
        _, counts = np.unique(codes, return_counts=True)
        top4_share = float(np.sort(counts)[-4:].sum()) / float(codes.size)
        if top4_share > 0.68:
            return False

        return True

    @staticmethod
    def _scene_terms(scene: Dict[str, Any]) -> Set[str]:
        stop = {
            "this", "that", "with", "from", "into", "over", "were", "have", "their", "about",
            "after", "before", "during", "would", "could", "should", "there", "where", "when",
            "what", "which", "while", "then", "than", "they", "them", "your", "very", "more",
            "most", "some", "only", "story", "scene", "history", "historical", "photo", "video",
            "footage", "archive", "archival", "image",
        }
        raw = " ".join([
            str(scene.get("visual_subject", "")),
            str(scene.get("video_query", "")),
            str(scene.get("wikimedia_query", "")),
            str(scene.get("wikipedia_article", "")),
            " ".join(map(str, scene.get("visual_keywords", []))),
        ])
        return {w.lower() for w in re.findall(r"[A-Za-z0-9]{3,}", raw) if w.lower() not in stop}

    @classmethod
    def _relevance_score(cls, candidate: Dict[str, Any], terms: Set[str]) -> int:
        text = (str(candidate.get("title", "")) + " " + str(candidate.get("desc", ""))).lower()
        if not terms or not text:
            return 0
        return sum(3 if term in str(candidate.get("title", "")).lower() else 1 for term in terms if term in text)

    @staticmethod
    def _validate_video_clip_frames(clip_path: Path) -> bool:
        if not clip_path.exists() or clip_path.stat().st_size < 4_000:
            return False

        sampled_arrays: List[np.ndarray] = []
        for t_sec in (0.05, 0.70, 1.40, 2.10):
            probe_jpg = clip_path.parent / f"_probe_{clip_path.stem}_{int(t_sec*100)}.jpg"
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-ss", f"{t_sec:.2f}", "-i", str(clip_path),
                        "-frames:v", "1", "-q:v", "4", str(probe_jpg),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                    check=False,
                )
                if not probe_jpg.exists():
                    return False
                with Image.open(probe_jpg) as im:
                    gray = im.convert("L").resize((160, 90), Image.Resampling.BILINEAR)
                    arr = np.asarray(gray, dtype=np.float32)
                probe_jpg.unlink(missing_ok=True)

                mean_luma = float(arr.mean())
                std_luma = float(arr.std())
                dark_ratio = float(np.mean(arr < 20.0))
                if mean_luma < 16.0 or std_luma < 7.0 or dark_ratio > 0.88:
                    return False
                sampled_arrays.append(arr)
            except Exception:
                probe_jpg.unlink(missing_ok=True)
                return False

        if len(sampled_arrays) < 3:
            return False

        diffs = [
            float(np.abs(sampled_arrays[i] - sampled_arrays[i - 1]).mean())
            for i in range(1, len(sampled_arrays))
        ]
        if float(np.mean(diffs)) < 0.25:
            return False
        return True

    def _download_and_validate_video_clip(self, cand: Dict[str, Any], dest_path: Path) -> Optional[Path]:
        url = cand.get("url", "")
        if not url or url in self.used_urls:
            return None
        if ABSOLUTE_REJECT_PATTERNS.search(f"{cand.get('title','')} {cand.get('desc','')} {url}"):
            return None

        cache_allowed = cand.get("source") != "US National Archives"
        cache_key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        cached_video = self.media_cache_dir / f"{cache_key}.mp4"
        if cache_allowed and cached_video.exists() and cached_video.stat().st_size > 4000:
            shutil.copy2(cached_video, dest_path)
            if self._validate_video_clip_frames(dest_path):
                self.used_urls.add(url)
                self.attributions.append(
                    f"• Video: {cand.get('title', 'Archival Video')[:70]} ({cand.get('license', 'Public Domain')}) — {cand.get('source_url', url)}"
                )
                return dest_path
            dest_path.unlink(missing_ok=True)

        dur = float(cand.get("duration", 12.0) or 12.0)
        if dur >= 25.0:
            seek_offsets = [round(dur * 0.35, 1), round(dur * 0.55, 1)]
        elif dur >= 8.0:
            seek_offsets = [round(dur * 0.25, 1), 1.2]
        else:
            seek_offsets = [0.3]

        for seek_s in seek_offsets:
            try:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{seek_s:.1f}",
                    "-t",
                    "3.8",
                    "-i",
                    url,
                    "-an",
                    "-vf",
                    "scale='min(1280,iw)':-2,fps=30",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    str(dest_path),
                ]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=35)
                if proc.returncode == 0 and self._validate_video_clip_frames(dest_path):
                    if cache_allowed:
                        shutil.copy2(dest_path, cached_video)
                    self.used_urls.add(url)
                    title = cand.get("title", "Archival Video")
                    lic = cand.get("license", "Public Domain / CC")
                    self.attributions.append(f"• Video: {title[:65]} ({lic}) — {cand.get('source_url', url)}")
                    return dest_path
                dest_path.unlink(missing_ok=True)
            except Exception:
                dest_path.unlink(missing_ok=True)
        return None

    def fetch_scene_asset(self, scene: dict, assets_dir: Path) -> Tuple[Path, str]:
        """
        v10.0 Scene-First Retrieval Pipeline:
        1. If `prefer_media == 'ai_image'` and budget (`<= 10`) is available -> generate bespoke 9:16 `MAI-Image-2.6-Flash` shot FIRST!
        2. If `prefer_media == 'video_clip'` -> search verified archival motion clips.
        3. Search unique, non-duplicate, non-diagram historical photographs (with 64-bit dHash perceptual deduplication).
        4. If no unique high-res archival photo passes quality & deduplication gates and AI budget remains -> generate `MAI-Image-2.6-Flash` shot!
        5. Distinct quadrant macro crop or designed editorial dossier plate as final fallback.
        """
        assets_dir.mkdir(parents=True, exist_ok=True)
        scene_id = int(scene.get("scene_id", 1))
        narration = str(scene.get("narration", "")).strip()
        prefer = str(scene.get("prefer_media", "ai_image"))
        can_use_ai = self.ai_images_used < self.max_ai_images and bool(self.foundry_endpoint and self.foundry_key)
        subject = str(scene.get("visual_subject") or "").strip()

        queries: List[str] = []
        for value in (
            scene.get("wikimedia_query"),
            scene.get("video_query"),
            subject,
            scene.get("wikipedia_article"),
            " ".join(map(str, scene.get("visual_keywords") or [])),
        ):
            value = str(value or "").strip()
            if value and value.lower() not in {x.lower() for x in queries}:
                queries.append(value)
        if not queries:
            queries = [" ".join(sorted(self._scene_terms(scene))[:5]) or self.topic]

        # ---------------------------------------------------------------------
        # 1) Bespoke 9:16 MAI-Image-2.6-Flash First when `prefer == "ai_image"`
        # ---------------------------------------------------------------------
        if prefer == "ai_image" and can_use_ai:
            prompt = str(scene.get("image_prompt") or subject or narration)
            ai_img = self._generate_foundry_image(prompt, assets_dir / f"scene_{scene_id:02d}_ai.jpg")
            if ai_img:
                self.ai_images_used += 1
                self.ai_generated_images.append(ai_img)
                self.downloaded_scene_images.append(ai_img)
                self.scene_audit_log.append({
                    "scene_id": scene_id,
                    "kind": f"AI RECONSTRUCTION ({self.ai_images_used}/{self.max_ai_images})",
                    "source": ai_img.name,
                    "narration": narration,
                    "source_url": "AI-generated; not archival",
                })
                return ai_img, "image"

        # ---------------------------------------------------------------------
        # 2) Try Authentic Archival Video Clip (when explicitly preferred)
        # ---------------------------------------------------------------------
        if prefer == "video_clip" and self.video_clips_used < 3:
            video_candidates: List[dict] = []
            for q in queries[:2]:
                video_candidates.extend(self.archive_finder.search(q, "video", limit=6))
            video_candidates = rank_candidates(video_candidates, scene)
            for candidate in video_candidates:
                if candidate.get("relevance_score", 0) < 2 or candidate.get("url") in self.used_urls:
                    continue
                clip_path = assets_dir / f"scene_{scene_id:02d}_archive.mp4"
                saved = self._download_and_validate_video_clip(candidate, clip_path)
                if saved:
                    self.video_clips_used += 1
                    title = candidate.get("title", "Archival video")
                    self.scene_audit_log.append({
                        "scene_id": scene_id,
                        "kind": "AUTHENTIC VIDEO",
                        "source": f"{candidate.get('source')}: {title[:70]}",
                        "narration": narration,
                        "source_url": candidate.get("source_url"),
                        "license": candidate.get("license"),
                        "relevance_score": candidate.get("relevance_score"),
                    })
                    return saved, "video"

        # ---------------------------------------------------------------------
        # 3) Search Multi-Archive Historical Photographs (Strict Deduplication & No Schematics)
        # ---------------------------------------------------------------------
        image_candidates: List[dict] = list(self.topic_article_images)
        scene_wp = str(scene.get("wikipedia_article") or "").strip()
        if scene_wp and scene_wp not in self.wikipedia_articles:
            image_candidates.extend(self.archive_finder.fetch_wikipedia_article_images(scene_wp))

        for q in queries[:3]:
            image_candidates.extend(self.archive_finder.search(q, "image", limit=10))

        image_candidates = rank_candidates(image_candidates, scene)
        image_dest = assets_dir / f"scene_{scene_id:02d}_archive.jpg"

        min_thresholds = (2,) if can_use_ai else (2, 1)
        for min_rel in min_thresholds:
            for candidate in image_candidates:
                if candidate.get("relevance_score", 0) < min_rel or candidate.get("url") in self.used_urls:
                    continue
                saved = self._download_and_register_image(
                    candidate.get("url", ""),
                    image_dest,
                    candidate.get("title", "Archival image"),
                    candidate.get("license", "Public Domain / CC"),
                    cache_allowed=candidate.get("source") != "US National Archives",
                )
                if saved:
                    self.archival_photos_used += 1
                    self.downloaded_scene_images.append(saved)
                    source = candidate.get("source", "Archive")
                    self.scene_audit_log.append({
                        "scene_id": scene_id,
                        "kind": "ARCHIVAL IMAGE",
                        "source": f"{source}: {candidate.get('title', '')[:70]}",
                        "narration": narration,
                        "source_url": candidate.get("source_url"),
                        "license": candidate.get("license"),
                        "relevance_score": candidate.get("relevance_score"),
                    })
                    return saved, "image"

        # ---------------------------------------------------------------------
        # 4) Gap-Filler AI Image (whenever archival search found no unique photo and budget remains)
        # ---------------------------------------------------------------------
        if can_use_ai:
            prompt = str(scene.get("image_prompt") or subject or narration)
            ai_img = self._generate_foundry_image(prompt, assets_dir / f"scene_{scene_id:02d}_ai.jpg")
            if ai_img:
                self.ai_images_used += 1
                self.ai_generated_images.append(ai_img)
                self.downloaded_scene_images.append(ai_img)
                self.scene_audit_log.append({
                    "scene_id": scene_id,
                    "kind": f"AI RECONSTRUCTION ({self.ai_images_used}/{self.max_ai_images})",
                    "source": ai_img.name,
                    "narration": narration,
                    "source_url": "AI-generated; not archival",
                })
                return ai_img, "image"

        # ---------------------------------------------------------------------
        # 5) Distinct Archival Macro Crop
        # ---------------------------------------------------------------------
        if self.downloaded_scene_images:
            src_photo = self.downloaded_scene_images[(scene_id - 1) % len(self.downloaded_scene_images)]
            macro_path = assets_dir / f"scene_{scene_id:02d}_macro.jpg"
            try:
                self._create_distinct_macro_shot(src_photo, macro_path, variant=scene_id)
                self.archival_photos_used += 1
                self.scene_audit_log.append({
                    "scene_id": scene_id,
                    "kind": "ARCHIVAL IMAGE (DETAIL CROP)",
                    "source": f"Detail crop from {src_photo.name}",
                    "narration": narration,
                    "source_url": "",
                    "license": "Public Domain / CC",
                })
                return macro_path, "image"
            except Exception:
                pass

        # ---------------------------------------------------------------------
        # 6) Designed Editorial Graphic Plate (final fallback)
        # ---------------------------------------------------------------------
        fallback_path = assets_dir / f"scene_{scene_id:02d}_editorial.jpg"
        self._create_archival_fallback_card(fallback_path, scene=scene, topic=self.topic)
        self.reframed_cuts_used += 1
        self.scene_audit_log.append({
            "scene_id": scene_id,
            "kind": "EDITORIAL GRAPHIC",
            "source": "Locally rendered editorial graphic",
            "narration": narration,
            "source_url": "",
            "license": "Original generated graphic",
        })
        return fallback_path, "image"

    def _download_and_register_image(
        self, img_url: str, dest_path: Path, title: str = "", license_name: str = "Public Domain / CC", cache_allowed: bool = True
    ) -> Optional[Path]:
        if not img_url or img_url in self.used_urls:
            return None
        if ABSOLUTE_REJECT_PATTERNS.search(f"{title} {img_url}") or DISQUALIFYING_PATTERNS.search(img_url):
            return None

        title_stem = self._normalize_title_stem(title)
        if title_stem and len(title_stem) >= 8 and title_stem in self.used_title_stems:
            return None

        try:
            cache_key = hashlib.sha256(img_url.encode("utf-8")).hexdigest()
            cached_image = self.media_cache_dir / f"{cache_key}.jpg"
            if cache_allowed and cached_image.exists() and cached_image.stat().st_size > 8000:
                shutil.copy2(cached_image, dest_path)
                content_bytes = dest_path.read_bytes()
            else:
                img_resp = self.session.get(img_url, timeout=25)
                img_resp.raise_for_status()
                content_bytes = img_resp.content

            digest = hashlib.md5(content_bytes).hexdigest()
            if digest in self.used_hashes:
                return None

            dest_path.write_bytes(content_bytes)
            with Image.open(dest_path) as im:
                w, h = im.size
                if w < 420 or h < 340:
                    dest_path.unlink(missing_ok=True)
                    return None
                rgb = im.convert("RGB")
                if not self._is_valid_photographic_image(rgb):
                    dest_path.unlink(missing_ok=True)
                    return None
                dhash = self._compute_dhash(rgb)
                if self._is_perceptual_duplicate(dhash):
                    dest_path.unlink(missing_ok=True)
                    return None

                rgb.save(dest_path, quality=95)
                if cache_allowed and not cached_image.exists():
                    shutil.copy2(dest_path, cached_image)

            self.used_urls.add(img_url)
            self.used_hashes.add(digest)
            self.used_dhashes.append(dhash)
            if title_stem and len(title_stem) >= 8:
                self.used_title_stems.add(title_stem)
            if title:
                clean_title = title.replace("File:", "")
                self.attributions.append(f"• {clean_title[:75]} ({license_name}) — {img_url}")
            return dest_path
        except Exception:
            dest_path.unlink(missing_ok=True)
            return None

    def _generate_foundry_image(self, prompt: str, dest_path: Path) -> Optional[Path]:
        try:
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
                f"{prompt}, ultra-sharp photorealistic historical documentary cinematography, dramatic rim lighting, "
                "razor-sharp focus on subject, 35mm Kodak archival film look, vertical 9:16 portrait composition, "
                "no blur, no text, no words, no watermarks"
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
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
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
                        rgb = im.convert("RGB")
                        self.used_dhashes.append(self._compute_dhash(rgb))
                        rgb.save(dest_path, quality=96)
                    return dest_path
            return None
        except Exception as exc:
            print(f"[MediaFetcher] Foundry {self.image_model} notice: {exc}")
            return None

    @classmethod
    def _create_distinct_macro_shot(cls, source_path: Path, dest_path: Path, variant: int = 1):
        with Image.open(source_path) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            crop_w = int(w * 0.62)
            crop_h = int(h * 0.62)
            quadrants = [
                (0, int(h * 0.08)),
                (w - crop_w, int(h * 0.14)),
                (int(w * 0.18), int(h * 0.30)),
                (0, max(0, h - crop_h)),
            ]
            ox, oy = quadrants[variant % len(quadrants)]
            cropped = rgb.crop((ox, oy, ox + crop_w, oy + crop_h))
            cropped = ImageEnhance.Contrast(cropped).enhance(1.12)
            cropped = ImageEnhance.Sharpness(cropped).enhance(1.15)
            cropped.save(dest_path, quality=95)

    @staticmethod
    def _create_archival_fallback_card(dest_path: Path, scene: Optional[Dict[str, Any]] = None, topic: str = "US History"):
        w, h = 1080, 1920
        scene = scene or {}
        yy, xx = np.mgrid[0:h, 0:w]
        t = yy.astype(np.float32) / max(1, h - 1)
        grain = ((xx * 17 + yy * 31) % 7).astype(np.int16) - 3
        base = np.stack([18 - 5 * t, 31 - 11 * t, 46 - 17 * t], axis=-1)
        arr = np.clip(base + grain[..., None], 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        d = ImageDraw.Draw(img)
        gold = (220, 174, 88)
        pale = (237, 239, 239)

        def font(size: int, bold: bool = False):
            paths = [
                "/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf" if bold else "/usr/share/fonts/truetype/montserrat/Montserrat-Medium.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
            for fp in paths:
                try:
                    return ImageFont.truetype(fp, size)
                except Exception:
                    pass
            return ImageFont.load_default()

        d.rectangle((70, 100, 1010, 1820), outline=(71, 87, 99), width=2)
        d.line((110, 260, 970, 260), fill=gold, width=5)
        d.text((112, 155), "HISTORICAL DOSSIER  //  CHRONICLE RECORD", font=font(27, True), fill=gold)

        cx, cy = 540, 740
        for rad, col, width in [(290, (49, 68, 83), 3), (220, (65, 82, 94), 2), (145, (85, 99, 106), 2)]:
            d.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), outline=col, width=width)
        d.ellipse((cx - 16, cy - 16, cx + 16, cy + 16), fill=gold)
        d.line((cx - 330, cy, cx + 330, cy), fill=(55, 73, 86), width=2)
        d.line((cx, cy - 330, cx, cy + 330), fill=(55, 73, 86), width=2)

        callout = str(scene.get("on_screen_callout") or "ARCHIVE RECORD").strip().upper()
        d.text((115, 1140), callout[:32], font=font(34, True), fill=gold)
        subject = str(scene.get("visual_subject") or topic).strip()

        def wrap(text: str, fnt, max_width: int):
            words = text.split()
            lines = []
            cur = ""
            for word in words:
                test = (cur + " " + word).strip()
                if d.textbbox((0, 0), test, font=fnt)[2] <= max_width:
                    cur = test
                else:
                    if cur:
                        lines.append(cur)
                    cur = word
            if cur:
                lines.append(cur)
            return lines

        y = 1215
        for line in wrap(subject.upper(), font(50, True), 820)[:3]:
            d.text((115, y), line, font=font(50, True), fill=pale)
            y += 70

        img.save(dest_path, quality=94)
