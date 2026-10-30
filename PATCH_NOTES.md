# Patch notes — cost-first US History Shorts automation

## Files to replace
Copy the files in this patch ZIP into the matching paths in the root of your existing repository, overwriting those files.

## What this patch changes
1. Paid AI image generation is OFF by default (`MAX_AI_IMAGES_PER_SHORT=0`).
2. The workflow-dispatch default and scheduled workflow fallback are both zero, so a missing secret cannot silently restore the old 4-image budget.
3. Scene 1 is no longer forced to use a paid AI image. An AI image is generated only when the scene explicitly requests `ai_image` and the budget is greater than zero.
4. If no suitable footage or archival photo is found, the pipeline creates a scene-specific local editorial card instead of reusing an unrelated AI/stock image.
5. Topic-approved video candidates must also match at least one meaningful word from the scene's own query/narration. Archival photo matching also rejects zero-overlap candidates.
6. Pexels results are disabled by default because the current code does not independently run its strict LLM approval step on each returned Pexels result. To opt in, set `ALLOW_PEXELS_UNVERIFIED=true` and review the footage.
7. Storyboard generation asks for credible research references and more careful handling of disputed or unsupported facts. Valid source URLs are added to the YouTube description and metadata.
8. YouTube privacy defaults to `unlisted` rather than `public`. Set the GitHub secret `YOUTUBE_PRIVACY_STATUS=public` only after reviewing output quality.

## Suggested GitHub Actions secrets / variables
- Keep existing Foundry text and Azure Speech secrets.
- Set `MAX_AI_IMAGES_PER_SHORT=0` (or omit it; zero is the default).
- Set `YOUTUBE_PRIVACY_STATUS=unlisted` during testing.
- Leave `ALLOW_PEXELS_UNVERIFIED` unset or `false` initially.

## Important
This is a targeted patch against the uploaded repository snapshot, not a replacement of your entire repository. It does not add new paid providers or require new API keys. The code was syntax-compiled and the local scene-fallback renderer was smoke-tested. A full live run still requires your API credentials, GitHub Actions environment, FFmpeg, and YouTube account.

The current pipeline still uses your Foundry text model for storyboard creation and media-candidate verification. This patch minimizes image-generation spend; it does not claim to eliminate text-model, Azure TTS, GitHub Actions, or YouTube API costs.
