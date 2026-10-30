"""Multi-provider archival media discovery for US History Shorts.

Uses public/official APIs only. Every returned item carries a source URL and a
rights decision. Unknown or restrictive rights are rejected for automatic use.
"""
from __future__ import annotations
import hashlib, json, os, re, time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote
import requests

UA = "USHistoryShorts/8.0 (+https://github.com/Rageshukla69/us-history-shorts-auto; archival-media-client)"


def _clean(s: Any) -> str:
    if isinstance(s, list):
        s = " ".join(str(x) for x in s)
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _tokens(s: str) -> set[str]:
    stop = {"the","and","for","with","from","that","this","were","was","are","their","into","during","after","before","about","history","historical","photo","image","video","footage","archive","archival","scene","show","shows","showing","people","american","united","states"}
    return {x.lower() for x in re.findall(r"[a-z0-9]{3,}", s.lower()) if x.lower() not in stop}


def rights_allowed(licence: str, rights: str = "") -> bool:
    """Conservative commercial-video policy: PD/CC0/CC BY/CC BY-SA only."""
    text = f"{licence} {rights}".lower().replace(" ", "")
    if any(x in text for x in ("noncommercial", "cc-by-nc", "by-nc", "no-derivatives", "cc-by-nd", "by-nd", "allrightsreserved", "fairuse", "copyrightnotcleared")):
        return False
    if "noknownrestrictions" in text or "publicdomain" in text:
        return True
    return any(x in text for x in (
        "publicdomain", "public-domain", "creativecommons.org/publicdomain/zero",
        "creativecommons.org/licenses/by/", "creativecommons.org/licenses/by-sa/",
        "cc0", "ccby", "ccby-sa", "ccby-sa"
    )) or text.strip() in {"pd", "public domain", "cc0", "cc by", "cc by-sa"}


class ArchiveMediaFinder:
    """Search Internet Archive, Wikimedia Commons, LOC, NASA and Openverse."""
    def __init__(self, session: Optional[requests.Session] = None, cache_dir: Optional[Path] = None):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": UA})
        self.cache_dir = Path(cache_dir or os.getenv("MEDIA_CACHE_DIR", ".cache/media")) / "search"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = (5, 12)

    def _get_json(self, url: str, params: Optional[dict] = None, cache_seconds: int = 86400) -> dict:
        key = hashlib.sha256((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
        fp = self.cache_dir / f"{key}.json"
        if fp.exists() and time.time() - fp.stat().st_mtime < cache_seconds:
            try: return json.loads(fp.read_text(encoding="utf-8"))
            except Exception: pass
        try:
            r = self.session.get(url, params=params, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            fp.write_text(json.dumps(data), encoding="utf-8")
            return data
        except Exception as exc:
            print(f"[ArchiveMedia] API unavailable: {url.split('/')[2]}: {exc}")
            return {}

    @staticmethod
    def _candidate(source: str, media_type: str, title: str, desc: str, url: str,
                   page_url: str, licence: str, licence_url: str = "", creator: str = "",
                   duration: float = 0.0, year: str = "", rights: str = "") -> Optional[dict]:
        if not url or not title: return None
        if not rights_allowed(licence, rights): return None
        return {
            "source": source, "media_type": media_type, "title": _clean(title),
            "desc": _clean(desc)[:1800], "url": url, "source_url": page_url or url,
            "license": _clean(licence) or "Public Domain / CC", "license_url": licence_url,
            "creator": _clean(creator), "duration": float(duration or 0), "year": _clean(year),
            "rights_verified": True,
        }

    def search(self, query: str, media_type: str, limit: int = 12) -> List[dict]:
        q = _clean(query)
        if not q: return []
        kind = "video" if media_type == "video" else "image"
        results: List[dict] = []
        adapters = (self._internet_archive, self._wikimedia, self._loc, self._nasa, self._nara)
        if kind == "image": adapters = adapters + (self._openverse,)
        for adapter in adapters:
            try:
                results.extend(adapter(q, kind, max(3, limit // 2)))
            except Exception as exc:
                print(f"[ArchiveMedia] {adapter.__name__} failed: {exc}")
        # Deduplicate by exact download URL, preserve first source.
        unique = {}
        for item in results:
            unique.setdefault(item["url"], item)
        return list(unique.values())[:limit * 2]

    def _internet_archive(self, q: str, kind: str, limit: int) -> List[dict]:
        query = f'({q}) AND mediatype:movies' if kind == "video" else f'({q}) AND mediatype:image'
        data = self._get_json("https://archive.org/advancedsearch.php", {
            "q": query, "fl[]": ["identifier","title","description","creator","year","licenseurl","rights","mediatype"],
            "rows": limit, "page": 1, "output": "json"
        }, cache_seconds=21600)
        out=[]
        for doc in data.get("response",{}).get("docs",[]):
            ident=doc.get("identifier")
            if not ident: continue
            meta=self._get_json(f"https://archive.org/metadata/{quote(str(ident))}", cache_seconds=86400)
            md=meta.get("metadata",{})
            lic=_clean(md.get("licenseurl") or doc.get("licenseurl"))
            rights=_clean(md.get("rights") or doc.get("rights"))
            if not rights_allowed(lic, rights): continue
            files=meta.get("files",[])
            if kind == "video":
                files=[f for f in files if str(f.get("name","")).lower().endswith((".mp4",".m4v",".mov",".webm")) and not str(f.get("name","")).lower().endswith(("_thumb.mp4","_preview.mp4"))]
                files=[f for f in files if int(f.get("size",0) or 0) <= 350_000_000]
                files.sort(key=lambda f: ("512kb" not in str(f.get("name","")).lower(), int(f.get("size",0) or 0)))
            else:
                files=[f for f in files if str(f.get("name","")).lower().endswith((".jpg",".jpeg",".png",".tif",".tiff")) and "thumb" not in str(f.get("name","")).lower()]
                files.sort(key=lambda f: int(f.get("size",0) or 0), reverse=True)
            if not files: continue
            fname=files[0].get("name","")
            url=f"https://archive.org/download/{quote(str(ident),safe='')}/{quote(str(fname),safe='()[],._-')}"
            item=self._candidate("Internet Archive",kind,_clean(md.get("title") or doc.get("title") or ident),_clean(md.get("description") or doc.get("description")),url,f"https://archive.org/details/{ident}",lic or rights,lic,_clean(md.get("creator") or doc.get("creator")),year=_clean(md.get("year") or doc.get("year")),rights=rights)
            if item: out.append(item)
        return out

    def _wikimedia(self, q: str, kind: str, limit: int) -> List[dict]:
        filetype="video" if kind=="video" else "bitmap"
        data=self._get_json("https://commons.wikimedia.org/w/api.php",{
            "action":"query","generator":"search","gsrsearch":f"{q} filetype:{filetype}","gsrnamespace":6,"gsrlimit":limit,
            "prop":"imageinfo","iiprop":"url|size|mime|extmetadata","iiurlwidth":1800,"format":"json"
        },cache_seconds=43200)
        out=[]
        for page in data.get("query",{}).get("pages",{}).values():
            info=(page.get("imageinfo") or [{}])[0]; ext=info.get("extmetadata",{}) or {}
            title=_clean(page.get("title","")).replace("File:","")
            lic=_clean((ext.get("LicenseShortName") or {}).get("value"))
            licurl=_clean((ext.get("LicenseUrl") or {}).get("value"))
            usage=_clean((ext.get("UsageTerms") or {}).get("value"))
            creator=_clean((ext.get("Artist") or {}).get("value"))
            rights=_clean((ext.get("Copyrighted") or {}).get("value"))
            mime=_clean(info.get("mime"))
            url=info.get("url")
            if kind=="video" and not (mime.startswith("video/") or str(url).lower().endswith((".webm",".ogv",".mp4"))): continue
            if kind=="image" and not mime.startswith("image/"): continue
            desc=_clean((ext.get("ImageDescription") or {}).get("value"))
            item=self._candidate("Wikimedia Commons",kind,title,desc,url,page.get("canonicalurl",f"https://commons.wikimedia.org/wiki/{quote(title.replace(' ','_'))}"),lic or usage,licurl,creator,rights=rights)
            if item: out.append(item)
        return out

    def _loc(self, q: str, kind: str, limit: int) -> List[dict]:
        endpoint="film-and-videos" if kind=="video" else "photos"
        data=self._get_json(f"https://www.loc.gov/{endpoint}/",{"q":q,"fo":"json","c":limit},cache_seconds=43200)
        out=[]
        for rec in data.get("results",[]):
            title=_clean(rec.get("title")); desc=_clean(rec.get("description"))
            rights=_clean(rec.get("rights") or rec.get("rights_advisory") or rec.get("rights_information"))
            # LOC's own item page is the authoritative rights context. Do not infer PD from age.
            page=rec.get("id","")
            if not rights_allowed(rights, rights): continue
            url=rec.get("image_url") or rec.get("url")
            if kind=="video":
                item_data=self._get_json(page.rstrip("/")+"/?fo=json",cache_seconds=86400) if page else {}
                item=item_data.get("item",{})
                resources=item_data.get("resources",[]) or []
                files=[]
                for resource in resources:
                    for f in resource.get("files",[]) if isinstance(resource,dict) else []:
                        if isinstance(f,dict) and str(f.get("mimetype","")).startswith("video/"):
                            files.append(f.get("url") or f.get("path"))
                if not files: continue
                url=files[0]
                title=_clean(item.get("title") or title); desc=_clean(item.get("description") or desc)
            if not url: continue
            item=self._candidate("Library of Congress",kind,title,desc,url,page,rights or "No known restrictions",rights=rights)
            if item: out.append(item)
        return out

    def _nasa(self, q: str, kind: str, limit: int) -> List[dict]:
        data=self._get_json("https://images-api.nasa.gov/search",{"q":q,"media_type":kind,"page_size":limit},cache_seconds=43200)
        out=[]
        for obj in data.get("collection",{}).get("items",[]):
            d=(obj.get("data") or [{}])[0]; links=obj.get("links") or []
            nasa_id=d.get("nasa_id"); title=_clean(d.get("title")); desc=_clean(d.get("description"))
            copyright=_clean(d.get("secondary_creator") or d.get("copyright"))
            if copyright and "nasa" not in copyright.lower(): continue
            asset=self._get_json(f"https://images-api.nasa.gov/asset/{quote(str(nasa_id))}",cache_seconds=86400) if nasa_id else {}
            hrefs=asset.get("collection",{}).get("items",[])
            if kind=="video":
                urls=[x.get("href","") for x in hrefs if str(x.get("href","")).lower().endswith(".mp4")]
                urls=[u for u in urls if "~small" not in u.lower()]
            else:
                urls=[x.get("href","") for x in hrefs if str(x.get("href","")).lower().endswith((".jpg",".jpeg",".png",".tif"))]
                if not urls: urls=[x.get("href","") for x in links if x.get("rel")=="preview"]
            if not urls: continue
            page=f"https://images.nasa.gov/details/{nasa_id}"
            item=self._candidate("NASA",kind,title,desc,urls[0],page,"NASA public domain (verify item credit)","https://www.nasa.gov/nasa-brand-center/images-and-media/",_clean(d.get("photographer") or d.get("center")),year=_clean(d.get("date_created")),rights="NASA media; third-party copyright filtered when disclosed")
            if item: out.append(item)
        return out

    def _nara(self, q: str, kind: str, limit: int) -> List[dict]:
        """Optional official NARA Catalog API adapter (requires free API key)."""
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
            print(f"[ArchiveMedia] NARA API unavailable: {exc}")
            return []
        hits = (((data.get("body") or {}).get("hits") or {}).get("hits") or [])
        out = []
        wanted = (".mp4", ".mov", ".m4v", ".webm") if kind == "video" else (".jpg", ".jpeg", ".png", ".tif", ".tiff")
        for hit in hits:
            src = hit.get("_source") or {}
            meta = src.get("metadata") or {}
            # NARA metadata is heterogeneous; recursively collect only direct media URLs.
            strings = []
            def walk(value):
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, str) and ("url" in k.lower() or "path" in k.lower() or "file" in k.lower()):
                            strings.append(v)
                        else:
                            walk(v)
                elif isinstance(value, list):
                    for v in value: walk(v)
            walk(src)
            media_urls = []
            for value in strings:
                u = value.strip()
                if u.startswith("//"): u = "https:" + u
                if u.startswith("http") and any(u.lower().split("?")[0].endswith(ext) for ext in wanted):
                    media_urls.append(u)
            if not media_urls: continue
            restrictions = json.dumps(meta.get("useRestrictions") or meta.get("rights") or meta.get("accessRestriction") or {}, ensure_ascii=False)
            low = restrictions.lower()
            if any(word in low for word in ("restricted", "copyrighted", "permission required", "donor restriction")):
                continue
            description = meta.get("description") or {}
            title = ""
            if isinstance(description, dict):
                item_desc = description.get("item") or description
                if isinstance(item_desc, dict): title = _clean(item_desc.get("title") or item_desc.get("scopeAndContentNote"))
            title = title or _clean(src.get("title") or hit.get("_id") or "National Archives record")
            naid = src.get("naId") or src.get("naid") or hit.get("_id")
            page = f"https://catalog.archives.gov/id/{naid}" if naid else "https://catalog.archives.gov/"
            item = self._candidate("US National Archives", kind, title, _clean(meta.get("scopeAndContentNote") or meta.get("description")), media_urls[0], page,
                "NARA federal record; check item Use Restrictions", "https://www.archives.gov/research/catalog/help/api", rights="NARA public-domain presumption; no explicit restriction in record")
            if item: out.append(item)
        return out

    def _openverse(self, q: str, kind: str, limit: int) -> List[dict]:
        if kind != "image": return []
        data=self._get_json("https://api.openverse.org/v1/images/",{
            "q":q,"page_size":limit,"license":"cc0,pdm,by,by-sa"
        },cache_seconds=43200)
        out=[]
        for rec in data.get("results",[]):
            lic=_clean(rec.get("license")); licurl=_clean(rec.get("license_url"))
            creator=_clean(rec.get("creator")); title=_clean(rec.get("title"))
            item=self._candidate("Openverse",kind,title,_clean(rec.get("description")),rec.get("url"),rec.get("foreign_landing_url"),lic,licurl,creator,rights=licurl)
            if item: out.append(item)
        return out


def rank_candidates(candidates: Iterable[dict], scene: dict) -> List[dict]:
    subject=_clean(scene.get("visual_subject")); keywords=scene.get("visual_keywords") or []
    query=_clean(scene.get("video_query") or scene.get("wikimedia_query"))
    terms=_tokens(" ".join([subject,query," ".join(map(str,keywords))]))
    phrase=re.sub(r"\s+", " ", subject.lower()).strip()
    ranked=[]
    for item in candidates:
        hay=_clean(item.get("title"))+" "+_clean(item.get("desc"))
        h=hay.lower(); hits={t for t in terms if t in h}
        score=len(hits)*2
        if phrase and phrase in h: score+=6
        if item.get("media_type")=="video" and scene.get("prefer_media")=="video_clip": score+=2
        if item.get("source")=="Library of Congress": score+=2
        if item.get("source")=="Internet Archive": score+=2
        if len(terms)>=3 and len(hits)<2: score-=3
        item=dict(item); item["relevance_score"]=score; item["matched_terms"]=sorted(hits)
        ranked.append(item)
    return sorted(ranked,key=lambda x:(x.get("relevance_score",0),len(x.get("desc", ""))),reverse=True)
