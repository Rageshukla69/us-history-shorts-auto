# Autonomous US History YouTube Shorts Engine (Every 12 Hours)

Tailored to work directly with your **existing Heroku Config Vars** (`AZURE_SPEECH_KEY`, `FOUNDRY_TEXT_ENDPOINT`, `FOUNDRY_ENDPOINT`, `MAI-Image-2.6-Flash`, `gpt-6-luna`, `YOUTUBE_TOKEN_JSON`) with **zero new accounts or API keys required**.

## Your Exact Secret Mapping (From Your Heroku App)
| Secret Name | What Value to Paste |
| :--- | :--- |
| `AZURE_SPEECH_KEY` | Copy directly from your Heroku `AZURE_SPEECH_KEY` |
| `AZURE_SPEECH_REGION` | `centralindia` (already pre-filled in Colab script!) |
| `AZURE_TTS_VOICE` | Pre-set to `en-US-BrianMultilingualNeural` (energetic young US male creator voice; automatically replaces `hi-IN-MadhurNeural` even if you copy your old var) |
| `BG_MUSIC_VOLUME` | `0.25` (already pre-filled) |
| `FOUNDRY_API_KEY` | Copy directly from your Heroku `FOUNDRY_API_KEY` |
| `FOUNDRY_ENDPOINT` | Copy directly from your Heroku `FOUNDRY_ENDPOINT` (`https://draftbot.services...`) |
| `FOUNDRY_IMAGE_MODEL` | `MAI-Image-2.6-Flash` (already pre-filled) |
| `FOUNDRY_TEXT_API_KEY` | Copy directly from your Heroku `FOUNDRY_TEXT_API_KEY` |
| `FOUNDRY_TEXT_ENDPOINT` | Copy directly from your Heroku `FOUNDRY_TEXT_ENDPOINT` (`https://vakilbot.services...`) |
| `FOUNDRY_TEXT_MODEL` | `gpt-6-luna` (already pre-filled) |
| `SHORTS_PER_DAY` | `2` (already pre-filled — runs every 12 hours) |
| `VOICE_SPEED_PERCENT` | `3` (already pre-filled — +3% crisp documentary pace) |
| `YOUTUBE_TOKEN_JSON` | Copy the full `{"token": "ya29...", ...}` JSON string from your Heroku `YOUTUBE_TOKEN_JSON` |

### Why Is There No Wikipedia API Key?
- **Wikipedia & Wikimedia Commons are 100% free public APIs**—they do **not** use API keys! `src/media_fetcher.py` queries them directly out of the box.
- For scenes that need custom close-ups or visual puns, the repo uses your existing **`MAI-Image-2.6-Flash`** model via `FOUNDRY_ENDPOINT` + `FOUNDRY_API_KEY`, so you **do not need a Pexels API key** (`PEXELS_API_KEY` is completely optional and can be left blank).
