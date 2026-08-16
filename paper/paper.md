---
title: "UNHUDDLE: neighborhood-aware signal reallocation for multiplex spatial proteomics"
tags:
  - Python
  - spatial proteomics
  - imaging mass cytometry
  - multiplex imaging
  - AnnData
authors:
  - name: Troy Noordenbos
    orcid: 0000-0000-0000-0000
    affiliation: "1, 2"
affiliations:
  - name: Stanford University School of Medicine
    index: 1
    ror: 00f54p054
  - name: Alizadeh Laboratory
    index: 2
date: 16 August 2026
bibliography: paper.bib
---

# Summary

UNHUDDLE (Uncovering Neighborhood Heterogeneity Using Deterministic Normalization and Local Equilibrium) is a Python pipeline for multiplex spatial proteomics. In densely packed tissue, neighboring cells share border pixels because of optical resolution limits, lateral bleed, and z-projection. Absolute segmentation then mixes neighbor signal into each cell's intensity, which blurs phenotypes and inflates apparent co-expression.

UNHUDDLE treats those shared pixels as a bookkeeping problem rather than a segmentation failure. It identifies membrane-adjacent interactions, estimates each neighbor's claim on the shared intensity, and reallocates border signal to the cells that most likely produced it. Optional cohort-level denoising then reduces residual noise measured in cell cores. Intensities can be normalized by total protein content, cell size, or housekeeper markers, after which UNHUDDLE writes Scanpy-compatible AnnData objects with morphology features, spatial coordinates, quality-control flags, and multiple intensity layers.

The software is intended for researchers working with imaging mass cytometry, multiplex immunofluorescence, and related highly multiplexed tissue assays. It runs from a command-line interface on `{marker}.ome.tiff` folders, can generate DeepCell-Mesmer masks when none are available, and includes a walkthrough plus demo data so that new users can reproduce a complete analysis without writing custom code.

# Statement of need

Spatial proteomics is increasingly used to map immune, stromal, and tumor neighborhoods in intact tissue. Downstream clustering, differential expression, and neighborhood statistics all depend on per-cell marker intensities. In packed regions those intensities are systematically contaminated: a T-cell membrane pixel overlapping a B-cell border is counted twice, once for each object. Existing workflows often clip membranes, use exclusion masks, or accept the mixed signal. Those heuristics discard information or leave neighbor bias in place.

UNHUDDLE is designed for experimental biologists and computational analysts who already have multiplex images and want a reproducible, documented path from FOV folders to an analysis-ready AnnData object. It addresses three practical needs:

1. **Border-aware quantification.** Shared pixels are reallocated using neighbor interactions rather than discarded.
2. **Cohort-aware denoising.** Optional noisecone or percentile denoising uses the full cohort to stabilize core intensities before reallocation weights are applied.
3. **End-to-end packaging.** Morphology, protein tables, QC plots, and an AnnData object are produced from one CLI, with optional DeepCell-Mesmer segmentation when masks are missing.

Related tools solve adjacent problems. IMC-Denoise focuses on pixel-level IMC artifacts [@Lu2023IMCDenoise]. PENGUIN provides graphical preprocessing for multiplex images [@Sequeira2024PENGUIN]. DeepCell Mesmer provides whole-cell segmentation [@Greenwald2022Mesmer]. Scanpy and AnnData provide the downstream analysis substrate [@Wolf2018Scanpy; @Virshup2024AnnData]. UNHUDDLE does not replace those tools; it sits between segmentation and single-cell analysis and contributes a deterministic reallocation model for neighbor-mixed border signal.

# State of the field

Most multiplex proteomics pipelines quantify cells by summing or averaging pixels inside a segmentation mask. CellProfiler, MCMICRO, and vendor platforms follow this pattern and then hand tables to clustering tools [@McQuin2018CellProfiler; @Schapiro2022MCMICRO]. When membranes are thick relative to cell diameter, that approach systematically mixes neighbors. Common workarounds include shrinking masks, excluding membrane pixels, or using only nuclear-proximal pixels. Those methods reduce bleed but also remove true membrane marker signal.

Pixel-level denoisers such as IMC-Denoise improve raw images before segmentation [@Lu2023IMCDenoise]. They do not model which neighboring cell owns a shared border pixel after objects have been defined. Deep learning segmenters such as Mesmer improve object boundaries [@Greenwald2022Mesmer], but even accurate masks still contain mixed pixels at contacts. Contribution to those codebases would not have produced a neighbor-reallocation layer or a cohort denoiser tied to UNHUDDLE's intensity accounting.

UNHUDDLE's distinct contribution is therefore not another segmenter or another clustering notebook. It is a documented, installable pipeline that (i) builds typed interaction graphs from cell and membrane masks, (ii) reallocates shared intensity with explicit conservation of signal, (iii) optionally denoises reallocation factors at cohort scale, and (iv) emits AnnData layers for original, unhuddled, and denoised intensities so that analysts can compare methods without rerunning image processing.

# Software design

The pipeline is organized as staged, FOV-parallel processing with a small number of cohort-wide steps.

**Inputs.** Each field of view is a folder of `{marker}.ome.tiff` files plus an optional segmentation mask. Nuclear markers are declared on the command line so chromatin signal can be separated from protein quantification.

**Per-FOV stages.** Mask processing builds cell, membrane, and membrane-exclusion masks. Feature extraction writes morphology and per-cell protein tables. Interaction computation records border contacts and background contacts. Reallocation then redistributes shared intensity using core (membrane-excluded) measurements as weights. The default compilation keeps original whole-cell sums and applies denoised or undenoised reallocation adjustments, which avoids silently throwing away core signal.

**Cohort stages.** Denoising, when enabled, fits a size-aware noise model (`noisecone`) or a percentile cutoff across the cohort. Normalization then scales markers using total protein, area, or user-specified housekeepers, with robust percentile scaling and a binarization fallback when dynamic range is insufficient. AnnData construction concatenates FOVs, attaches spatial masks, and runs QC filters for low intensity and local density.

This design trades a fully interactive GUI for a reproducible CLI and filesystem contract. Parallelism is isolated to FOV-independent stages so multiprocessing remains simple. Cohort steps are sequential because denoising and scaling need all cells. Keeping original, unhuddled, and denoised layers in one AnnData object is an explicit product decision: users can audit the correction instead of trusting a single transformed matrix.

# Research impact statement

UNHUDDLE is already packaged as a public GitHub repository with a walkthrough, bundled demo FOV, optional extended demo-data download, and a CLI that produces QC figures plus an AnnData object. The v1.0.0 release adds an OSI MIT license, citation metadata (`CITATION.cff`), and this short paper so the software can be archived with a DOI and submitted for independent review.

Near-term research use is in multiplex tissue studies where neighbor mixing is a known confounder, including tonsil and tumor immune microenvironments processed in the originating laboratory. Community-readiness signals include: (i) a single install path (`pip install -e .` today; `pip install unhuddle` as packaging matures), (ii) compatibility aliases after the denoise-repo merge, (iii) Scanpy/AnnData interoperability, and (iv) optional DeepCell and PENGUIN integrations rather than a closed stack. Benchmark helper scripts in the documentation support cell-level before/after intensity comparison, which is the evidence format reviewers and collaborators typically request.

# AI usage disclosure

Generative AI assistance was used to draft repository maintenance files for the v1.0.0 public release, including license/citation metadata, this JOSS paper draft, and release documentation. Pipeline algorithms, scientific claims, and CLI behavior were implemented and checked by the author. The paper text was reviewed against the software's actual CLI flags, output layout, and README walkthrough. Any remaining factual errors are the author's responsibility.

# Acknowledgements

UNHUDDLE originated in work on high-dimensional tissue profiling in the Alizadeh laboratory at Stanford School of Medicine. DeepCell Mesmer [@Greenwald2022Mesmer] and PENGUIN [@Sequeira2024PENGUIN] are third-party tools; users should cite those authors when those optional modules are used.

# References
