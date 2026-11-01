# Autonomous US History YouTube Shorts Studio (v9.0 — HeyGen HyperFrames + 10 MAI-Image-2.6-Flash Edition)

A hands-free, cloud-native YouTube Shorts studio that automatically researches, scripts, voices, animates via **HeyGen HyperFrames (`HTML + CSS + GSAP + HyperShader`)**, renders in `1080x1920` (30 FPS), and publishes **2 high-retention US History Shorts per day** using **GitHub Actions**, **Microsoft Foundry (`gpt-6-luna` + `MAI-Image-2.6-Flash`)**, **Azure Speech (`en-US-BrianMultilingualNeural`)**, and **Multi-Archive Public Domain Media**.

## What Is New in v9.0 (HyperFrames + 10 MAI Images Edition)
1. **Official HeyGen HyperFrames Composition Engine (`src/hyperframes_composer.py` + `package.json`)**:
   - Authors a complete, spec-compliant `1080x1920` HyperFrames project (`output/hyperframes_project/index.html`, `preview.html`, `DESIGN.md`) for every Short.
   - Uses a deterministic paused **GSAP 3.14.2** timeline (`window.__timelines["main"] = tl`) and WebGL **HyperShader** transitions (`cinematic-zoom`) at the midpoint act break.
   - Includes **4 simultaneous mid-scene motion layers** on every cut: optical blur-to-sharp entrance snap, continuous Ken Burns scale/pan, SVG timeline stroke draw + rotating reticle (`strokeDashoffset`), live animated year/stat counters (`tabular-nums`), and 3-color word-synced kinetic captions.
   - Runs `npx hyperframes lint` and `npx hyperframes render` (with an automatic hybrid FFmpeg dual-camera fallback so CI runs never fail on headless runners).
2. **Up to 10 `MAI-Image-2.6-Flash` Cinema Reconstructions per Short (`MAX_AI_IMAGES_PER_SHORT = 10`)**:
   - Generates bespoke vertical `9:16` (`768x1344` / `1024x1792`) historical cinema shots for dramatic hooks, close-ups, and twists while still blending in authentic archival video clips and photographs from Wikipedia, Wikimedia Commons, Prelinger Archives, Library of Congress, and NASA.
