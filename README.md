# Autonomous US History YouTube Shorts Engine (v3.0 — Documentary Edition)

Tailored to work directly with your **existing Heroku Config Vars** (`AZURE_SPEECH_KEY`, `FOUNDRY_TEXT_ENDPOINT`, `FOUNDRY_ENDPOINT`, `MAI-Image-2.6-Flash`, `gpt-6-luna`, `YOUTUBE_TOKEN_JSON`) with **zero new accounts or API keys required**.

## What Was Upgraded in v3.0
1. **100% Boundary-Safe Word-Synced Captions (`src/caption_builder.py`)**
   - Switched ASS subtitle engine from `WrapStyle: 2` (which disabled wrapping and clipped words at the left/right edges) to `WrapStyle: 0` with `90px` horizontal safe margins, character-aware 1-to-2 line chunking, and `62pt` Montserrat typography.
   - Merged standalone punctuation tokens (`,`, `.`, `;`, `:`) directly onto the preceding word so floating spaces like `FRANCE ,` or `RUBBER ;` never appear.
   - Deduplicated consecutive identical top callout badges so badges don't blink repeatedly.
2. **MAI-Image Credit Conservation Guardrail (`src/media_fetcher.py` & `MAX_AI_IMAGES_PER_SHORT`)**
   - Added a strict `MAX_AI_IMAGES_PER_SHORT` cap (default `2`, configurable down to `1` or `0` in GitHub Secrets) so `MAI-Image-2.6-Flash` is only called at most 1–2 times per Short, slashing your Foundry image credit usage by **80%–90%**.
   - Pre-warms a verified historical image pool from the topic's main Wikipedia articles and Wikimedia Commons categories and uses zero-credit documentary detail crops when needed.
3. **Zero Random Celebrities or Tourist Signs (`src/media_fetcher.py` & `src/foundry_llm.py`)**
   - Strips generic metaphor/material words (`rubber`, `looks real`, `illusion`, `actors`, `secret`) and anchors every Wikimedia/Wikipedia lookup to the core historical subject and era.
   - Added a strict regex blacklist against modern celebrities (`Prince Harry`, `Invictus Games`, `red carpet`, `press conference`) and modern tourist plaques (`memorial plaque`, `information board`, `wayside marker`), plus URL/MD5 deduplication across all scenes.
4. **Dark Mysterious Documentary Score & Zero "Zoop/Whoosh" SFX (`src/sfx_generator.py` & `src/video_composer.py`)**
   - Completely removed `whoosh`, `pop`, `boom`, and `shutter` transition sound effects from every cut.
   - Synthesizes a 68-second stereo dark mysterious D-minor documentary score (`assets/sfx/mystery_bgm.wav`) featuring deep sub-bass pulses, haunting cello/synth tension chords, eerie high harmonic shimmer, and a 60 BPM suspense heartbeat.
   - Uses FFmpeg `sidechaincompress` to automatically duck the mysterious background music underneath the narrator's voiceover and swell gently during pauses.
   - Frames landscape/square archival photographs over a frosted dark 9:16 canvas with top/bottom readability vignettes so subjects are never awkwardly chopped off.

## Your Exact Secret Mapping (From Your Heroku App)
| Secret Name | What Value to Paste |
| :--- | :--- |
| `AZURE_SPEECH_KEY` | Copy directly from your Heroku `AZURE_SPEECH_KEY` |
| `AZURE_SPEECH_REGION` | `centralindia` (already pre-filled in workflow) |
| `AZURE_TTS_VOICE` | Pre-set to `en-US-BrianMultilingualNeural` (automatically replaces `hi-IN-MadhurNeural`) |
| `BG_MUSIC_VOLUME` | `0.28` (dark mysterious documentary score volume before voiceover auto-ducking) |
| `FOUNDRY_API_KEY` | Copy directly from your Heroku `FOUNDRY_API_KEY` |
| `FOUNDRY_ENDPOINT` | Copy directly from your Heroku `FOUNDRY_ENDPOINT` (`https://draftbot.services...`) |
| `FOUNDRY_IMAGE_MODEL` | `MAI-Image-2.6-Flash` (already pre-filled) |
| `MAX_AI_IMAGES_PER_SHORT` | `2` (caps MAI-Image calls per video to save your credits; set to `1` or `0` to spend even less) |
| `FOUNDRY_TEXT_API_KEY` | Copy directly from your Heroku `FOUNDRY_TEXT_API_KEY` |
| `FOUNDRY_TEXT_ENDPOINT` | Copy directly from your Heroku `FOUNDRY_TEXT_ENDPOINT` (`https://vakilbot.services...`) |
| `FOUNDRY_TEXT_MODEL` | `gpt-6-luna` (already pre-filled) |
| `SHORTS_PER_DAY` | `2` (runs every 12 hours) |
| `VOICE_SPEED_PERCENT` | `3` (+3% crisp documentary pace) |
| `YOUTUBE_TOKEN_JSON` | Copy the full `{"token": "ya29...", ...}` JSON string from your Heroku `YOUTUBE_TOKEN_JSON` |
