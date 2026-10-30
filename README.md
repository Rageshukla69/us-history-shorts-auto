# US History Shorts Automation — Documentary Edition

This repository builds vertical history Shorts using a cost-first media pipeline. It is designed to prefer real, attributable historical media and to fail closed rather than publish unrelated or empty visuals.

## Core behaviour
- Paid image generation defaults to **0** (`MAX_AI_IMAGES_PER_SHORT=0`). It is only used when explicitly enabled and the storyboard marks a scene as requiring a reconstruction.
- Scene-level relevance matching is applied before using archival photos or videos. Topic-level approval alone is not enough.
- When no source asset is found, the renderer uses a clearly labelled editorial graphic, never a fake archival photo.
- If more than 30% of scenes have no suitable source media, the pre-upload quality gate blocks publication.
- The technical QC checks portrait dimensions, audio presence, duration, file size and sustained black frames. Human review is still required.
- Source/credit details are written into `output/metadata.json` and YouTube description where available.
- Workflow caches `.cache/media` to reduce repeated media downloads.

## Required GitHub Secrets
Keep your existing secrets; do not paste secret values into source files.

| Secret | Purpose |
|---|---|
| `FOUNDRY_TEXT_ENDPOINT`, `FOUNDRY_TEXT_API_KEY` | Storyboard generation |
| `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION` | Azure voiceover (Edge TTS fallback) |
| `YOUTUBE_TOKEN_JSON` | YouTube upload |
| `FOUNDRY_ENDPOINT`, `FOUNDRY_API_KEY` | Optional image generation |
| `PEXELS_API_KEY` | Optional stock-video search (disabled unless `ALLOW_PEXELS_STOCK=true`) |

Optional variables: `MAX_AI_IMAGES_PER_SHORT=0`, `YOUTUBE_PRIVACY_STATUS=unlisted`, `BG_MUSIC_VOLUME=0.18`, `VOICE_SPEED_PERCENT=2`. Pexels stock clips are opt-in via `ALLOW_PEXELS_STOCK=true`; they are not labelled archival footage.

## Safe first run
1. Run the GitHub Actions workflow manually with `dry_run=true` and `max_ai_images=0`.
2. Download `rendered-short-*` and inspect it from beginning to end.
3. Inspect `output/metadata.json`, `output/quality_report.json`, and the Actions summary.
4. Only after manual review, run with `dry_run=false`. Scheduled runs default to preview-only. To enable scheduled uploads, add repository secret `DRY_RUN=false`; remove that secret or set it to `true` to pause scheduled publishing.

## Important limitations
Free archives do not guarantee that every obscure scene has usable motion footage. The system must not pretend otherwise. A history short may use a mix of authentic moving footage, period photographs, maps and clearly labelled editorial graphics. A human editor must verify historical accuracy, licensing and final visual quality before public release.

## Project layout
- `src/foundry_llm.py`: story/script generation and candidate verification
- `src/media_fetcher.py`: scene-aware asset selection, image/video validation and fallbacks
- `src/video_composer.py`: 1080x1920 rendering, camera motion and audio mix
- `src/caption_builder.py`: word-synchronised captions
- `src/quality_control.py`: fail-closed pre-upload technical gate
- `.github/workflows/shorts_12h_pipeline.yml`: scheduled/manual workflow


## v8 multi-archive media engine

The scene-level media resolver searches official/public endpoints from Internet Archive, Wikimedia Commons, the Library of Congress, NASA Image and Video Library, and Openverse. It ranks results against each scene's concrete visual subject and keywords. A source is only auto-eligible when the returned item carries an explicit permissive rights basis (Public Domain, CC0, CC BY, or CC BY-SA). Non-commercial, no-derivatives, fair-use, and unknown-rights records are excluded from automatic use. This is a conservative automation filter, not legal advice; verify individual rights before monetized publication.

Media priority: rights-cleared moving footage -> rights-cleared historical still -> budgeted AI reconstruction -> clearly labelled original editorial graphic. AI image generation is a hard per-video cap and accepts zero. It never fabricates an archival source.

`MIN_AUTHENTIC_MEDIA_RATIO` defaults to `0.30`. Preview renders can still be produced below this ratio, but an actual upload is blocked until enough scenes use authentic, rights-cleared source media. Set `DRY_RUN=true` for safe testing.

Optional: set `NARA_API_KEY` (request one from NARA) to enable the official US National Archives Catalog adapter. The NARA API has usage terms and rate limits; its records can contain rare third-party restrictions, so the adapter rejects records with restriction indicators and preserves the item record URL. The software must not imply NARA endorsement.

Run `python scripts/diagnose_media_sources.py` to inspect which archives return rights-eligible results for representative subjects. It performs searches only and does not download or publish assets.
