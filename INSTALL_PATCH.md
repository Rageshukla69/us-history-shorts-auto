# Install the multi-archive media upgrade

Recommended: replace the full repository contents with this repository snapshot, preserving your GitHub Secrets. Ensure hidden `.github/` is included.

Patch-only: replace `src/archive_media.py`, `src/media_fetcher.py`, `src/quality_control.py`, `src/main.py`, `.env.example`, `README.md`, `MEDIA_SOURCES.md`, and `.github/workflows/shorts_12h_pipeline.yml`; add `scripts/diagnose_media_sources.py` and `tests/`.

Keep `MAX_AI_IMAGES_PER_SHORT=3` if you want at most three paid image generations per Short. Use `0` for none. For scheduled runs, add a GitHub Actions repository Variable named `MAX_AI_IMAGES_PER_SHORT` with value `3`; the manual Run workflow input overrides it. Optionally add the `NARA_API_KEY` GitHub Secret after requesting a key from NARA; this enables another high-value US history archive. The cap is enforced in Python as well as passed by Actions.

Run locally: `python -m unittest discover -s tests -v`. In GitHub Actions, run manually with `dry_run=true` first. Review `output/metadata.json`, `output/quality_report.json`, and the rendered MP4 artifact.

Rights policy: the downloader only auto-selects items with an explicit Public Domain, CC0, CC BY, or CC BY-SA indication. Rights and personality/trademark restrictions can still vary by item and jurisdiction; inspect the source record before monetized publication.
