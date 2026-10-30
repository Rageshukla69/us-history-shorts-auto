"""
Uploads the rendered 1080x1920 History Short directly to YouTube via YouTube Data API v3
using your existing `YOUTUBE_TOKEN_JSON` secret from Heroku (`{"token": "ya29...", "refresh_token": "...", ...}`).
"""
import json
import os
from pathlib import Path
from typing import List, Optional
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

TOKEN_URI = "https://oauth2.googleapis.com/token"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def _load_youtube_credentials() -> Credentials:
    raw_json = os.getenv("YOUTUBE_TOKEN_JSON", "").strip()
    if raw_json:
        info = json.loads(raw_json)
        creds = Credentials(
            token=info.get("token") or info.get("access_token"),
            refresh_token=info.get("refresh_token"),
            token_uri=info.get("token_uri", TOKEN_URI),
            client_id=info.get("client_id"),
            client_secret=info.get("client_secret"),
            scopes=info.get("scopes") or SCOPES,
        )
        if creds.refresh_token and creds.client_id and creds.client_secret:
            creds.refresh(Request())
        return creds

    # Fallback if individual env vars were set
    client_id = os.getenv("YOUTUBE_CLIENT_ID", "").strip()
    client_secret = os.getenv("YOUTUBE_CLIENT_SECRET", "").strip()
    refresh_token = os.getenv("YOUTUBE_REFRESH_TOKEN", "").strip()
    if not (client_id and client_secret and refresh_token):
        raise RuntimeError(
            "Missing YOUTUBE_TOKEN_JSON in GitHub Secrets! Copy your YOUTUBE_TOKEN_JSON string from Heroku."
        )
    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri=TOKEN_URI,
        client_id=client_id,
        client_secret=client_secret,
        scopes=SCOPES,
    )
    creds.refresh(Request())
    return creds


def upload_short_to_youtube(
    video_path: Path,
    title: str,
    description: str,
    tags: List[str],
    attributions: List[str],
) -> Optional[str]:
    if os.getenv("DRY_RUN", "false").lower() == "true":
        print("[YouTubeUploader] DRY_RUN is enabled; skipping YouTube API upload.")
        return "dry_run_id"

    creds = _load_youtube_credentials()
    youtube = build("youtube", "v3", credentials=creds)

    if "#shorts" not in title.lower():
        title = f"{title[:88]} #Shorts"

    full_description = description.strip() + "\n\n#Shorts #History #USHistory #Documentary\n"
    if attributions:
        unique_attr = list(dict.fromkeys(attributions))[:12]
        full_description += "\nArchival Media Credits (Wikimedia Commons):\n" + "\n".join(unique_attr)

    body = {
        "snippet": {
            "title": title[:100],
            "description": full_description[:4900],
            "tags": tags[:25] if tags else ["history", "shorts", "ushistory", "documentary"],
            "categoryId": "27",  # Education
            "defaultLanguage": "en-US",
            "defaultAudioLanguage": "en-US",
        },
        "status": {
            "privacyStatus": os.getenv("YOUTUBE_PRIVACY_STATUS", "unlisted").strip(),
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        str(video_path),
        mimetype="video/mp4",
        chunksize=1024 * 1024 * 8,
        resumable=True,
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"[YouTubeUploader] Upload progress: {int(status.progress() * 100)}%")

    video_id = response.get("id")
    print(f"[YouTubeUploader] Successfully published US History Short! https://youtube.com/shorts/{video_id}")
    return video_id
