# UNHUDDLE v1.0.0 release, archive, and registry checklist

This file is the operator checklist for the public v1.0.0 launch. GitHub tagging can be done from this repository. Zenodo DOI minting and Dutch Research Software Directory (RSD) registration require your logged-in accounts.

## 1. GitHub tag and release

Already prepared in this branch:

- MIT `LICENSE`
- `CITATION.cff`
- version `1.0.0` in `pyproject.toml`
- JOSS draft in `paper/paper.md` and `paper/paper.bib`
- `.zenodo.json` metadata for GitHub–Zenodo harvesting

Create the GitHub Release from tag `v1.0.0` (done by the release automation in this session if the push succeeded). Confirm at:

- https://github.com/noordenbos/Unhuddle/releases/tag/v1.0.0

## 2. Archive on Zenodo and get a DOI

Recommended path: GitHub Releases → Zenodo harvesting.

1. Sign in at [https://zenodo.org](https://zenodo.org) with the GitHub account that owns `noordenbos/Unhuddle`.
2. Open [https://zenodo.org/account/settings/github/](https://zenodo.org/account/settings/github/).
3. Flip the switch **on** for `noordenbos/Unhuddle`.
4. If v1.0.0 already exists, click **Sync now** / re-send the GitHub release webhook, or create a tiny patch release (`v1.0.1`) after enabling the hook so Zenodo receives an event.
5. Open the new Zenodo record and copy:
   - **Version DOI** (cites this exact release)
   - **Concept DOI** (always resolves to the latest version; use this in RSD)
6. Paste the version DOI into `CITATION.cff` (`doi:` field) and the README citation block, then tag `v1.0.1` if you want the archived source to contain the DOI.

Until that webhook is enabled, no DOI can be minted from here.

## 3. Register in the Dutch Research Software Directory

Production directory: [https://research-software-directory.org](https://research-software-directory.org)  
Docs: [Adding software](https://research-software-directory.org/documentation/users/adding-software/)  
Access: [How to get access](https://research-software-directory.org/documentation/users/getting-access/)

### 3a. Get an account (if you do not have one)

Email `rsd@esciencecenter.nl` with:

- Name: Troy Noordenbos
- Affiliation: Stanford University School of Medicine (and Dutch affiliation if applicable)
- ORCID: *(add yours)*
- Example content: https://github.com/noordenbos/Unhuddle

Alternatively, if your Dutch institute has enabled RSD in SURFconext, sign in with institute credentials.

### 3b. Create the software page

After sign-in: **+ → New Software**.

Paste:

- **Name:** UNHUDDLE
- **Short description:** Neighborhood-aware signal reallocation and normalization for multiplex spatial proteomics.
- **Get started URL:** https://github.com/noordenbos/Unhuddle#--unhuddle-tutorial
- **Repository URL:** https://github.com/noordenbos/Unhuddle
- **Software DOI:** the Zenodo *concept* DOI (preferred) or the v1.0.0 version DOI
- **License:** MIT
- **Keywords:** spatial proteomics, imaging mass cytometry, multiplex imaging, AnnData, Python
- **Contact person:** Troy Noordenbos

Then add organisations (Stanford; Dutch institute if relevant) and a mention of the JOSS preprint/submission once it has a URL.

## 4. Submit to JOSS

JOSS needs a public Git repository that contains `paper/paper.md` (this repo now does).

1. Preview the PDF locally if you have Docker:

   ```bash
   docker run --rm \
     --volume "$PWD/paper":/data \
     openjournals/inara:latest \
     -o pdf paper.md
   ```

2. Replace the placeholder ORCID `0000-0000-0000-0000` in `paper/paper.md` with your real ORCID.
3. Submit at [https://joss.theoj.org/papers/new](https://joss.theoj.org/papers/new) with:
   - Software repository: `https://github.com/noordenbos/Unhuddle`
   - Paper location: `paper/paper.md` on default branch after this PR merges
   - Suggested editor/reviewers: optional
4. Keep the review issue open; JOSS review is public on GitHub.

## Suggested software page blurb (RSD / Zenodo)

UNHUDDLE reallocates shared border-pixel intensity among neighboring cells in densely packed multiplex tissue images, then optionally denoises, normalizes, and exports a Scanpy-compatible AnnData object with QC. It is aimed at imaging mass cytometry and multiplex immunofluorescence users who need reproducible per-cell quantification without discarding membrane signal.
