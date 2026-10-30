# Media source registry

This project uses direct, documented search interfaces rather than scraping random stock sites.

| Provider | Media | Integration | Automatic rights policy |
|---|---|---|---|
| Internet Archive | Historical film and stills | Advanced Search + item metadata | Requires item-level Public Domain / CC0 / CC BY / CC BY-SA rights metadata |
| Wikimedia Commons | Photos, maps, documents, some video | MediaWiki API + image metadata | Requires item-level Public Domain / CC0 / CC BY / CC BY-SA licence |
| Library of Congress | Photos and film/video | Official JSON API | Requires a clear rights statement, including “No known restrictions on publication” |
| NASA Image and Video Library | Space/science/aviation photos and video | Search + asset manifest API | NASA public-domain presumption, reject disclosed third-party copyright; retain NASA credit |
| Openverse | Openly licensed still images | Openverse Images API | CC0 / PDM / CC BY / CC BY-SA only; source landing page retained |
| US National Archives | Digitised US federal records | Official Catalog API v2; optional `NARA_API_KEY` | NARA federal-record public-domain presumption, reject explicit restriction indicators; item-level review remains necessary |

## Source-selection rules

1. Search the actual scene subject, not just the whole video topic.
2. Prefer moving historical footage for physical action; use photographs for people, places, objects and events.
3. Rank candidates against the scene's concrete subject and keywords.
4. Reject candidates with poor subject overlap, unsupported media type, or restrictive/unknown rights.
5. Record title, creator where available, source page, licence and licence URL in the scene audit and video description.
6. Never describe an AI reconstruction or editorial graphic as archival evidence.

## Rights and use notes

- CC BY and CC BY-SA require attribution; CC BY-SA may impose share-alike obligations on adaptations.
- CC BY-NC and CC BY-ND are excluded from automatic monetized-video use.
- “No known restrictions” is a rights statement, not a universal legal warranty.
- NASA material can include third-party content; follow the individual item credit and NASA usage guidance.
- NARA's Catalog API requires a key, has rate limits, and asks API users not to cache/store Catalog API response content. The adapter therefore does not put NARA source files or responses into the reusable media cache. NARA requires the notice: “This product uses the National Archives Catalog API but is not endorsed or certified by the National Archives and Records Administration.”
- This software's automated rights filter is conservative guidance, not legal advice. Review the original item record before monetized publication.

## Open-source project inspiration

- [FootageFlow](https://github.com/xcslys99/FootageFlow): multi-provider search, fallback and source separation.
- [Open Archives](https://github.com/HarperZ9/open-archives): rights-aware asset receipts, provenance and credits.
- [Wikimedia Commons API](https://www.mediawiki.org/wiki/API:Main_page): documented media discovery and metadata.
- [Openverse](https://github.com/WordPress/openverse): openly licensed media discovery.
