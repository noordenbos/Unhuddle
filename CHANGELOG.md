# Changelog

## [1.0.0] - 2026-08-16

First public tagged release of the canonical UNHUDDLE repository.

- Integrate the denoise-pipeline rewrite into [noordenbos/Unhuddle](https://github.com/noordenbos/Unhuddle)
- Ship a single installable package (`unhuddle`) with CLI `unhuddle` and compatibility alias `unhuddle-denoise`
- Add cohort-level denoising (`noisecone` default; `percentile` available)
- Export Scanpy-compatible AnnData objects with QC flags and optional DeepCell-Mesmer mask generation
- Add OSI MIT license, `CITATION.cff`, and a JOSS paper draft under `paper/`
