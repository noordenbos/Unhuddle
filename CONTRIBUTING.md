# Contributing to UNHUDDLE

Thank you for helping improve UNHUDDLE. This repository is the canonical public project at [https://github.com/noordenbos/Unhuddle](https://github.com/noordenbos/Unhuddle).

## Development setup

```bash
git clone https://github.com/noordenbos/Unhuddle.git
cd Unhuddle
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The primary CLI is `unhuddle`. The `unhuddle-denoise` command remains as a compatibility alias.

## Smoke test

The bundled demo FOV is enough to check that the pipeline still runs:

```bash
unhuddle \
  --base_path demodata \
  --output_base_path results/dev_smoke \
  --nuclear_markers DNA1 DNA2 HistoneH3 \
  --create_nuclear_mask \
  --max_workers 1 \
  --create_adata \
  --no_denoise
```

A successful run writes `results/dev_smoke/adata_objects/adata1.h5ad`.

## Code style

- Format with Black (`line-length = 100`)
- Sort imports with isort (Black profile)
- Prefer small, reviewable pull requests against `main`

## Reporting issues

Please include:

- The exact `unhuddle` command
- Python version and operating system
- The log file under `{output_base_path}/logs/`
- A description of expected versus observed behavior

Do not attach unpublished patient images unless you have permission to share them.
