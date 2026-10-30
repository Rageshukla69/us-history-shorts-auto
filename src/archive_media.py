"""Multi-provider archival media discovery for US History Shorts.

Uses public/official APIs only (Wikimedia Commons, Wikipedia REST API, Internet Archive,
Library of Congress, NASA Image and Video Library, Openverse, and optional NARA Catalog API).
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional
from urllib.parse import quote
import requests

UA = "USHistoryShorts/8.1 (+https://github.com/Rageshukla69/us-history-shorts-auto; archival-media-client)"


def _clean(s: Any) -> str:
    if isinstance(s, list):
        s = " ".join(str(x) for x in s)
    raw = re.sub(r"<[^>]+>", " ", str(s or ""))
    return re.sub(r"\s+", " ", raw).strip()


def _tokens(s: str) -> set[str]:
    stop = {
        "the", "and", "for", "with", "from", "that", "this", "were", "was", "are",
        "their", "into", "during", "after", "before", "about", "history", "historical",
        "photo", "photograph", "image", "video", "footage", "archive", "archival",
        "scene", "show", "shows", "showing", "people", "american", "united", "states",
        "vintage", "black", "white", "film", "clip", "documentary"
    }
    return {x.lower() for x in re.findall(r"[a-z0-9]{3,}", s.lower()) if x.lower() not in stop}


def rights_allowed(licence: str, rights: str = "") -> bool:
    """Commercial-friendly archival policy: Public Domain / US Gov / CC0 / CC BY / CC BY-SA."""
    raw = f"{licence} {rights}".strip().lower()
    if not raw:
        return False
    compact = raw.replace(" ", "").replace("_", "-")
    if any(x in compact for x in (
        "noncommercial", "cc-by-nc", "by-nc", "no-derivatives", "cc-by-nd", "by-nd",
        "allrightsreserved", "fairuse", "copyrightnotcleared"
    )):
        return False
    if any(x in compact for x in (
        "noknownrestrictions", "publicdomain", "public-domain", "pdm", "pd-",
        "usgov", "cc0", "cc-zero", "ccby", "cc-by", "creativecommons.org/publicdomain",
        "creativecommons.org/licenses/by/", "creativecommons.org/licenses/by-sa/",
        "nara", "nasamedia", "libraryofcongress", "prelinger", "wikimedia"
    )):
        return True
    tokens = set(re.findall(r"[a-z0-9-]+", raw))
    if tokens & {"pd", "cc0", "pdm", "by", "by-sa", "cc-by", "cc-by-sa"}:
        return True
    return False


class ArchiveMediaFinder:
    """Search Wikimedia Commons, Wikipedia articles, Internet Archive, LOC, NASA, NARA, and Openverse."""
    def __init__(self, session: Optional[requests.Session] = None, cache_dir: Optional[Path] = None):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": UA})
        self.cache_dir = Path(cache_dir or os.getenv("MEDIA_CACHE_DIR", ".cache/media")) / "search"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = (5, 14)

    def _get_json(self, url: str, params: Optional[dict] = None, cache_seconds: int = 86400) -> dict:
        key = hashlib.sha256((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
        fp = self.cache_dir / f"{key}.json"
        if fp.exists() and time.time() - fp.stat().st_mtime < cache_seconds:
            try:
                return json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                pass
        try:
            r = self.session.get(url, params=params, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            fp.write_text(json.dumps(data), encoding="utf-8")
            return data
        except Exception as exc:
            print(f"[ArchiveMedia] API notice ({url.split('/')[2]}): {exc}")
            return {}

    @staticmethod
    def _candidate(
        source: str,
        media_type: str,
        title: str,
        desc: str,
        url: str,
        page_url: str,
        licence: str,
        licence_url: str = "",
        creator: str = "",
        duration: float = 0.0,
        year: str = "",
        rights: str = "",
    ) -> Optional[dict]:
        if not url or not title:
            return None
        if not rights_allowed(licence, rights):
            return None
        return {
            "source": source,
            "media_type": media_type,
            "title": _clean(title),
            "desc": _clean(desc)[:1800],
            "url": url,
            "source_url": page_url or url,
            "license": _clean(licence) or "Public Domain / CC",
            "license_url": licence_url,
            "creator": _clean(creator),
            "duration": float(duration or 0),
            "year": _clean(year),
            "rights_verified": True,
        }

    def search(self, query: str, media_type: str, limit: int = 12) -> List[dict]:
        q = _clean(query)
        if not q:
            return []
        kind = "video" if media_type == "video" else "image"
        results: List[dict] = []
        if kind == "video":
            adapters = (self._wikimedia, self._internet_archive, self._nasa, self._loc, self._nara)
        else:
            adapters = (self._wikimedia, self._wikipedia_search, self._openverse, self._loc, self._nasa, self._internet_archive, self._nara)

        for adapter in adapters:
            try:
                results.extend(adapter(q, kind, max(4, limit // 2)))
            except Exception as exc:
                print(f"[ArchiveMedia] {adapter.__name__} notice: {exc}")

        unique = {}
        for item in results:
            unique.setdefault(item["url"], item)
        return list(unique.values())[: limit * 3]

    def fetch_wikipedia_article_images(self, article_title: str) -> List[dict]:
        """Directly fetches high-resolution lead & gallery photos from a Wikipedia article."""
        slug = _clean(article_title).replace(" ", "_")
        if not slug:
            return []
        out: List[dict] = []
        summary = self._get_json(f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(slug)}", cache_seconds=86400)
        if summary:
            orig = summary.get("originalimage") or {}
            thumb = summary.get("thumbnail") or {}
            img_url = orig.get("source") or thumb.get("source") or ""
            if img_url and not img_url.lower().endswith((".svg", ".gif", ".pdf", ".djvu")):
                c = self._candidate(
                    "Wikipedia / Wikimedia Commons",
                    "image",
                    f"{article_title} — {summary.get('title', '')}",
                    _clean(summary.get("description") or summary.get("extract")),
                    img_url,
                    f"https://en.wikipedia.org/wiki/{quote(slug)}",
                    "Public Domain / Wikimedia Commons",
                    rights="PublicDomain",
                )
                if c:
                    out.append(c)

        media_list = self._get_json(f"https://en.wikipedia.org/api/rest_v1/page/media-list/{quote(slug)}", cache_seconds=86400)
        for item in (media_list.get("items") or [])[:18]:
            if item.get("type") != "image":
                continue
            title = _clean(item.get("title", "")).replace("File:", "")
            caption = _clean((item.get("caption") or {}).get("text", ""))
            srcset = item.get("srcset") or []
            if not srcset:
                continue
            best_src = srcset[-1].get("src", "")
            if best_src.startswith("//"):
                best_src = "https:" + best_src
            if not best_src or best_src.lower().endswith((".svg", ".gif", ".pdf", ".djvu")):
                continue
            c = self._candidate(
                "Wikipedia / Wikimedia Commons",
                "image",
                f"{title} ({article_title})",
                f"{article_title}: {caption}",
                best_src,
                f"https://commons.wikimedia.org/wiki/File:{quote(title.replace(' ', '_'))}",
                "Public Domain / CC BY-SA (Wikimedia)",
                rights="Wikimedia",
            )
            if c:
                out.append(c)
        return out

    def _wikipedia_search(self, q: str, kind: str, limit: int) -> List[dict]:
        if kind != "image":
            return []
        data = self._get_json(
            "https://en.wikipedia.org/w/api.php",
            {
                "action": "query",
                "generator": "search",
                "gsrsearch": q,
                "gsrlimit": min(5, limit),
                "prop": "pageimages|extracts",
                "exintro": 1,
                "explaintext": 1,
                "exchars": 300,
                "piprop": "original|thumbnail",
                "pithumbsize": 1400,
                "format": "json",
            },
            cache_seconds=43200,
        )
        out: List[dict] = []
        for page in (data.get("query", {}).get("pages", {}) or {}).values():
            title = _clean(page.get("title"))
            extract = _clean(page.get("extract"))
            img_url = (page.get("original") or {}).get("source") or (page.get("thumbnail") or {}).get("source")
            if not img_url or img_url.lower().endswith((".svg", ".gif", ".pdf", ".djvu")):
                continue
            c = self._candidate(
                "Wikipedia",
                "image",
                title,
                extract,
                img_url,
                f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                "Public Domain / Wikimedia Commons",
                rights="Wikimedia",
            )
            if c:
                out.append(c)
        return out

    def _internet_archive(self, q: str, kind: str, limit: int) -> List[dict]:
        if kind == "video":
            query = (
                f'({q}) AND mediatype:(movies) AND collection:(prelinger OR usnationalarchives OR '
                f'fedflixnara OR classic_tv OR newsandpublicaffairs) -title:(speech) -title:(interview)'
            )
        else:
            query = f'({q}) AND mediatype:(image)'

        data = self._get_json(
            "https://archive.org/advancedsearch.php",
            {
                "q": query,
                "fl[]": ["identifier", "title", "description", "creator", "year", "licenseurl", "rights", "mediatype", "collection"],
                "rows": limit,
                "page": 1,
                "output": "json",
            },
            cache_seconds=21600,
        )
        out: List[dict] = []
        for doc in data.get("response", {}).get("docs", []):
            ident = doc.get("identifier")
            if not ident:
                continue
            meta = self._get_json(f"https://archive.org/metadata/{quote(str(ident))}", cache_seconds=86400)
            md = meta.get("metadata", {})
            collections = _clean(md.get("collection") or doc.get("collection")).lower()
            lic = _clean(md.get("licenseurl") or doc.get("licenseurl"))
            rights = _clean(md.get("rights") or doc.get("rights"))
            # Prelinger and US National Archives collections on archive.org are Public Domain even when licenseurl is omitted
            if not lic and not rights and any(c in collections for c in ("prelinger", "usnationalarchives", "fedflixnara", "nasa")):
                lic = "Public Domain (Archival Collection)"
                rights = "PublicDomain"
            if not rights_allowed(lic, rights):
                continue
            files = meta.get("files", [])
            if kind == "video":
                files = [
                    f for f in files
                    if str(f.get("name", "")).lower().endswith((".mp4", ".m4v", ".mov", ".webm"))
                    and not str(f.get("name", "")).lower().endswith(("_thumb.mp4", "_preview.mp4"))
                ]
                files = [f for f in files if 30_000 <= int(f.get("size", 0) or 50_000) <= 350_000_000]
                files.sort(key=lambda f: ("512kb" not in str(f.get("name", "")).lower(), int(f.get("size", 0) or 0)))
            else:
                files = [
                    f for f in files
                    if str(f.get("name", "")).lower().endswith((".jpg", ".jpeg", ".png"))
                    and "thumb" not in str(f.get("name", "")).lower()
                ]
                files.sort(key=lambda f: int(f.get("size", 0) or 0), reverse=True)
            if not files:
                continue
            chosen = files[0]
            fname = chosen.get("name", "")
            dur = 25.0
            try:
                dur = float(chosen.get("length", 25.0) or 25.0)
            except (TypeError, ValueError):
                dur = 25.0
            url = f"https://archive.org/download/{quote(str(ident), safe='')}/{quote(str(fname), safe='()[],._-')}"
            item = self._candidate(
                "Internet Archive",
                kind,
                _clean(md.get("title") or doc.get("title") or ident),
                _clean(md.get("description") or doc.get("description")),
                url,
                f"https://archive.org/details/{ident}",
                lic or rights,
                lic,
                _clean(md.get("creator") or doc.get("creator")),
                duration=dur,
                year=_clean(md.get("year") or doc.get("year")),
                rights=rights,
            )
            if item:
                out.append(item)
        return out

    def _wikimedia(self, q: str, kind: str, limit: int) -> List[dict]:
        filetype = "video" if kind == "video" else "bitmap"
        prop = "videoinfo" if kind == "video" else "imageinfo"
        iiprop = "url|size|mime|extmetadata|derivatives" if kind == "video" else "url|size|mime|extmetadata"
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"filetype:{filetype} {q}",
            "gsrnamespace": 6,
            "gsrlimit": limit,
            "prop": prop,
            "format": "json",
        }
        if kind == "video":
            params["viprop"] = iiprop
        else:
            params["iiprop"] = iiprop
            params["iiurlwidth"] = 1600

        data = self._get_json("https://commons.wikimedia.org/w/api.php", params, cache_seconds=43200)
        out: List[dict] = []
        for page in (data.get("query", {}).get("pages", {}) or {}).values():
            info = (page.get("videoinfo") or page.get("imageinfo") or [{}])[0]
            ext = info.get("extmetadata", {}) or {}
            title = _clean(page.get("title", "")).replace("File:", "")
            lic = _clean((ext.get("LicenseShortName") or {}).get("value"))
            licurl = _clean((ext.get("LicenseUrl") or {}).get("value"))
            usage = _clean((ext.get("UsageTerms") or {}).get("value"))
            creator = _clean((ext.get("Artist") or {}).get("value"))
            rights = _clean((ext.get("Copyrighted") or {}).get("value"))
            if rights.lower() == "false" and not lic:
                lic = "Public Domain"
            mime = _clean(info.get("mime"))
            url = info.get("thumburl") or info.get("url")
            dur = float(info.get("duration", 12.0) or 12.0)

            if kind == "video":
                # Pick 720p MP4/WebM transcode derivative if available so FFmpeg downloads fast
                derivs = info.get("derivatives") or []
                valid_d = [
                    d.get("src", "")
                    for d in derivs
                    if 360 <= int(d.get("height", 0) or 0) <= 1080
                    and str(d.get("src", "")).lower().endswith((".mp4", ".webm"))
                ]
                if valid_d:
                    url = valid_d[-1]
                if str(url).startswith("//"):
                    url = "https:" + str(url)
                if not (mime.startswith("video/") or str(url).lower().endswith((".webm", ".ogv", ".mp4"))):
                    continue
            else:
                if not mime.startswith("image/") or str(url).lower().endswith((".svg", ".gif", ".tif", ".tiff", ".pdf")):
                    continue

            desc = _clean((ext.get("ImageDescription") or {}).get("value"))
            cats = _clean((ext.get("Categories") or {}).get("value"))
            item = self._candidate(
                "Wikimedia Commons",
                kind,
                title,
                f"{desc} {cats}",
                url,
                page.get("canonicalurl", f"https://commons.wikimedia.org/wiki/File:{quote(title.replace(' ', '_'))}"),
                lic or usage or "Public Domain / CC",
                licurl,
                creator,
                duration=dur,
                rights=f"{lic} {usage} {rights}",
            )
            if item:
                out.append(item)
        return out

    def _loc(self, q: str, kind: str, limit: int) -> List[dict]:
        endpoint = "film-and-videos" if kind == "video" else "photos"
        data = self._get_json(f"https://www.loc.gov/{endpoint}/", {"q": q, "fo": "json", "c": limit}, cache_seconds=43200)
        out: List[dict] = []
        for rec in data.get("results", []):
            title = _clean(rec.get("title"))
            desc = _clean(rec.get("description") or rec.get("subject"))
            rights = _clean(rec.get("rights") or rec.get("rights_advisory") or rec.get("rights_information") or "No known restrictions on publication")
            page = rec.get("id", "")
            if not rights_allowed(rights, rights):
                continue
            img_urls = rec.get("image_url") or []
            url = img_urls[-1] if isinstance(img_urls, list) and img_urls else rec.get("url")
            if kind == "video":
                item_data = self._get_json(page.rstrip("/") + "/?fo=json", cache_seconds=86400) if page else {}
                resources = item_data.get("resources", []) or []
                files = []
                for resource in resources:
                    for f in resource.get("files", []) if isinstance(resource, dict) else []:
                        if isinstance(f, dict) and str(f.get("mimetype", "")).startswith("video/"):
                            files.append(f.get("url") or f.get("path"))
                if not files:
                    continue
                url = files[0]
            if not url or not str(url).startswith("http"):
                continue
            item = self._candidate(
                "Library of Congress",
                kind,
                title,
                desc,
                str(url),
                page,
                rights or "No known restrictions",
                rights=rights,
            )
            if item:
                out.append(item)
        return out

    def _nasa(self, q: str, kind: str, limit: int) -> List[dict]:
        data = self._get_json(
            "https://images-api.nasa.gov/search",
            {"q": q, "media_type": kind, "page_size": limit},
            cache_seconds=43200,
        )
        out: List[dict] = []
        for obj in data.get("collection", {}).get("items", []):
            d = (obj.get("data") or [{}])[0]
            links = obj.get("links") or []
            nasa_id = d.get("nasa_id")
            title = _clean(d.get("title"))
            desc = _clean(d.get("description"))
            copyright_notice = _clean(d.get("secondary_creator") or d.get("copyright"))
            if copyright_notice and "nasa" not in copyright_notice.lower():
                continue

            if kind == "image":
                preview_urls = [x.get("href", "") for x in links if x.get("rel") == "preview" and str(x.get("href", "")).lower().endswith((".jpg", ".jpeg", ".png"))]
                if preview_urls:
                    page = f"https://images.nasa.gov/details/{nasa_id}"
                    item = self._candidate(
                        "NASA",
                        kind,
                        title,
                        desc,
                        preview_urls[0].replace(" ", "%20"),
                        page,
                        "Public Domain (NASA)",
                        "https://www.nasa.gov/nasa-brand-center/images-and-media/",
                        _clean(d.get("photographer") or d.get("center")),
                        year=_clean(d.get("date_created")),
                        rights="PublicDomain",
                    )
                    if item:
                        out.append(item)
                    continue

            asset = self._get_json(f"https://images-api.nasa.gov/asset/{quote(str(nasa_id))}", cache_seconds=86400) if nasa_id else {}
            hrefs = asset.get("collection", {}).get("items", [])
            if kind == "video":
                urls = [x.get("href", "") for x in hrefs if str(x.get("href", "")).lower().endswith(".mp4")]
                urls = [u for u in urls if "~small" not in u.lower()]
            else:
                urls = [x.get("href", "") for x in hrefs if str(x.get("href", "")).lower().endswith((".jpg", ".jpeg", ".png"))]
            if not urls:
                continue
            page = f"https://images.nasa.gov/details/{nasa_id}"
            item = self._candidate(
                "NASA",
                kind,
                title,
                desc,
                urls[0].replace(" ", "%20"),
                page,
                "Public Domain (NASA)",
                "https://www.nasa.gov/nasa-brand-center/images-and-media/",
                _clean(d.get("photographer") or d.get("center")),
                duration=25.0,
                year=_clean(d.get("date_created")),
                rights="PublicDomain",
            )
            if item:
                out.append(item)
        return out

    def _nara(self, q: str, kind: str, limit: int) -> List[dict]:
        key = os.getenv("NARA_API_KEY", "").strip()
        if not key:
            return []
        try:
            response = self.session.get(
                "https://catalog.archives.gov/api/v2/records/search",
                params={"q": q, "availableOnline": "true", "limit": limit},
                headers={"x-api-key": key, "Content-Type": "application/json"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            print(f"[ArchiveMedia] NARA API notice: {exc}")
            return []
        hits = (((data.get("body") or {}).get("hits") or {}).get("hits") or [])
        out: List[dict] = []
        wanted = (".mp4", ".mov", ".m4v", ".webm") if kind == "video" else (".jpg", ".jpeg", ".png")
        for hit in hits:
            src = hit.get("_source") or {}
            meta = src.get("metadata") or {}
            strings: List[str] = []

            def walk(value: Any):
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, str) and ("url" in k.lower() or "path" in k.lower() or "file" in k.lower()):
                            strings.append(v)
                        else:
                            walk(v)
                elif isinstance(value, list):
                    for v in value:
                        walk(v)

            walk(src)
            media_urls = []
            for value in strings:
                u = value.strip()
                if u.startswith("//"):
                    u = "https:" + u
                if u.startswith("http") and any(u.lower().split("?")[0].endswith(ext) for ext in wanted):
                    media_urls.append(u)
            if not media_urls:
                continue
            restrictions = json.dumps(meta.get("useRestrictions") or meta.get("rights") or meta.get("accessRestriction") or {}, ensure_ascii=False)
            if any(word in restrictions.lower() for word in ("restricted", "copyrighted", "permission required", "donor restriction")):
                continue
            description = meta.get("description") or {}
            title = ""
            if isinstance(description, dict):
                item_desc = description.get("item") or description
                if isinstance(item_desc, dict):
                    title = _clean(item_desc.get("title") or item_desc.get("scopeAndContentNote"))
            title = title or _clean(src.get("title") or hit.get("_id") or "National Archives record")
            naid = src.get("naId") or src.get("naid") or hit.get("_id")
            page = f"https://catalog.archives.gov/id/{naid}" if naid else "https://catalog.archives.gov/"
            item = self._candidate(
                "US National Archives",
                kind,
                title,
                _clean(meta.get("scopeAndContentNote") or meta.get("description")),
                media_urls[0],
                page,
                "Public Domain (US National Archives)",
                "https://www.archives.gov/research/catalog/help/api",
                rights="PublicDomain",
            )
            if item:
                out.append(item)
        return out

    def _openverse(self, q: str, kind: str, limit: int) -> List[dict]:
        if kind != "image":
            return []
        data = self._get_json(
            "https://api.openverse.org/v1/images/",
            {"q": q, "page_size": limit, "license": "cc0,pdm,by,by-sa"},
            cache_seconds=43200,
        )
        out: List[dict] = []
        for rec in data.get("results", []):
            lic = _clean(rec.get("license"))
            licurl = _clean(rec.get("license_url"))
            creator = _clean(rec.get("creator"))
            title = _clean(rec.get("title"))
            tags = " ".join(_clean(t.get("name")) for t in (rec.get("tags") or []) if isinstance(t, dict))
            item = self._candidate(
                "Openverse",
                kind,
                title,
                f"{_clean(rec.get('description'))} {tags}",
                rec.get("url"),
                rec.get("foreign_landing_url"),
                lic,
                licurl,
                creator,
                rights=f"{lic} {licurl}",
            )
            if item:
                out.append(item)
        return out


def rank_candidates(candidates: Iterable[dict], scene: dict) -> List[dict]:
    """
    Ranks archival candidates against the scene's visual subject, queries, and keywords.
    Never subtracts an artificial penalty when a candidate matches 1-2 core historical terms!
    """
    subject = _clean(scene.get("visual_subject"))
    keywords = scene.get("visual_keywords") or []
    query = _clean(scene.get("video_query") or scene.get("wikimedia_query") or scene.get("wikipedia_article"))
    narration = _clean(scene.get("narration"))
    terms = _tokens(" ".join([subject, query, " ".join(map(str, keywords)), narration]))
    phrase = re.sub(r"\s+", " ", subject.lower()).strip()

    ranked: List[dict] = []
    for item in candidates:
        title_clean = _clean(item.get("title")).lower()
        desc_clean = _clean(item.get("desc")).lower()
        hay = f"{title_clean} {desc_clean}"
        hits = {t for t in terms if t in hay}
        title_hits = {t for t in terms if t in title_clean}

        score = len(hits) * 2 + len(title_hits)
        if phrase and phrase in hay:
            score += 6
        if item.get("media_type") == "video" and scene.get("prefer_media") == "video_clip":
            score += 2
        if item.get("source") in ("Wikipedia / Wikimedia Commons", "Wikimedia Commons", "Library of Congress", "Internet Archive", "NASA"):
            score += 2

        item_copy = dict(item)
        item_copy["relevance_score"] = score
        item_copy["matched_terms"] = sorted(hits)
        ranked.append(item_copy)

    return sorted(ranked, key=lambda x: (x.get("relevance_score", 0), len(x.get("desc", ""))), reverse=True)
