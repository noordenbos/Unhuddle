<h2>
  UNHUDDLE
  <img src="assets/images/logo.png" alt="logo" width="90" align="left"/>
</h2>

**Uncovering Neighborhood Heterogeneity Using Deterministic Normalization and Local Equilibrium**

<br>UNHUDDLE is an algorithm designed to resolve signal in densely packed tissue regions — or "cell huddles" — in multiplex spatial proteomics, where traditional absolute segmentation introduces 'neighbor noise' and blur the phenotypic signal.

On the cell to cell borderpixels, shared signal is observed due to 1 resolution issues, 2 lateral bleed and 3 z-projection. Unhuddle knows the cell's neighbors, measures their mean intensity and reallocates the bordersignal to the rightful owner. 

By identifying stable, broadly expressed "normalisation markers" and performing per-cell normalization, UNHUDDLE enables more accurate within-cell-type comparison of functional markers (e.g., checkpoint proteins), even in spatially crowded microenvironments.

## 🚀 Getting Started with UNHUDDLE

### 📥 1. Clone the Repository
via https
```bash
git clone https://github.com/tbee05/unhuddle.git
cd unhuddle
```
via ssh
```bash
git clone git@github.com:tbee05/unhuddle.git
cd unhuddle
```
### 📦 2. Set Up a Virtual Environment
Using venv:
```bash
python -m venv unhuddle
source unhuddle/bin/activate      # On Windows: unhuddle\Scripts\activate
```
or conda:
```bash
conda create -n unhuddle -y
conda activate unhuddle
conda install pip
```
### 🛠️ 3. Install UNHUDDLE in Editable Mode
```bash
pip install -e .
```
This installs unhuddle as a CLI tool available from anywhere in your terminal.

### ✅ 4. Verify Installation
```bash
unhuddle --help
```
Should print a list of CLI arguments and options.

### 🗂️ 5. Check Input Requirements

The base input directory should contain one folder per FOV:

```
base_path/
├── FOV1/
│   ├── CD3.ome.tiff
│   ├── CD20.ome.tiff
│   └── ...
├── FOV2/
│   ├── CD3.ome.tiff
│   └── ...
```
Each FOV folder should contain:
- Denoised marker images (`{marker}.ome.tiff`, shape: `H x W`, dtype: `float32/64`)
- Optionally: a segmentation mask (`*_mask_0.tiff`, shape:   `H x W` or `Z x H x W`, dtype: `uint16`)  
  If a mask is not provided, one can be generated using `--create_deepcell_mask`.

🎯**protip:**  
contain the patientID in the FOV-name: "{patientID}_{FOVnumber}".  
NB do not use underscores within patientID or FOVnumber.
 
example for the first fov of patient 23:
```
base_path/
├── P23_1/
│   ├── CD3.ome.tiff
```
### 🧪 6. Run the Pipeline on Included Demo Data
linux:
```bash
unhuddle \
  --base_path demodata \
  --output_base_path results/unhuddle_output \
  --nuclear_markers DNA1 DNA2 HistoneH3 \
  --normalisation_markers CD20 CD68 CD11b CD11c CD8a CD3 CD7 CD45RA CD45RO CD15 CD163 Vimentin CD31 CD14 \
  --create_nuclear_mask \
  --max_workers 1 \
  --create_adata
```
windows powershell
```powershell
unhuddle `
  --base_path demodata `
  --output_base_path results\unhuddle_output `
  --nuclear_markers DNA1 DNA2 HistoneH3 `
  --normalisation_markers CD20 CD68 CD11b CD11c CD8a CD3 CD7 CD45RA CD45RO CD15 CD163 Vimentin CD31 CD14 `
  --create_nuclear_mask `
  --max_workers 1 `
  --create_adata
```


## 🌐 DeepCell Integration
UNHUDDLE can upload overlays to DeepCell.org using Selenium and Firefox with the geckodriver — no GUI required.
This enables an end-to-end pipeline, from pixeldata to Unhuddled single cell profiles.
### 🛠️ Manual Setup: Firefox + GeckoDriver

If Firefox is not available on your system, follow these steps to install both Firefox and GeckoDriver locally into `~/tools` or `%USERPROFILE%\tools`.

---

<details>
<summary><strong>🐧 Linux Instructions</strong></summary>

### 🦊 1. Install Firefox (Portable)
```bash
mkdir -p "$HOME/tools"
cd "$HOME/tools"
wget "https://download.mozilla.org/?product=firefox-latest&os=linux64&lang=en-US" -O firefox.tar.bz2
tar -xjf firefox.tar.bz2
```

Firefox will now be at:
```bash
$HOME/tools/firefox/firefox
```

### 🧭 2. Install GeckoDriver
```bash
cd "$HOME/tools"
wget https://github.com/mozilla/geckodriver/releases/download/v0.35.0/geckodriver-v0.35.0-linux64.tar.gz
tar -xvzf geckodriver-*.tar.gz
chmod +x geckodriver
```

✅ GeckoDriver path:
```bash
$HOME/tools/geckodriver
```

</details>

---

<details>
<summary><strong>🍎 macOS Instructions</strong></summary>

### 🦊 1. Install Firefox
Download from the official website:

🔗 https://www.mozilla.org/en-US/firefox/new/

Or for a portable install:
- Drag `Firefox.app` into a custom folder, e.g.:
```bash
$HOME/tools/Firefox.app
```

### 🧭 2. Install GeckoDriver
```bash
cd "$HOME/tools"
curl -LO https://github.com/mozilla/geckodriver/releases/download/v0.35.0/geckodriver-v0.35.0-macos.tar.gz
tar -xvzf geckodriver-*.tar.gz
chmod +x geckodriver
```

✅ GeckoDriver path:
```bash
$HOME/tools/geckodriver
```

</details>

---

<details>
<summary><strong>🪟 Windows Instructions (PowerShell)</strong></summary>

### 🦊 1. Install Firefox
Download from the official site:

🔗 https://www.mozilla.org/en-US/firefox/new/

Use the custom install option to place it in:
```
%USERPROFILE%\tools\Firefox
```

### 🧭 2. Install GeckoDriver
```powershell
$toolsDir = "$env:USERPROFILE\tools"
New-Item -ItemType Directory -Force -Path $toolsDir
Set-Location -Path $toolsDir

Invoke-WebRequest -Uri "https://github.com/mozilla/geckodriver/releases/download/v0.35.0/geckodriver-v0.35.0-win64.zip" -OutFile "$toolsDir\geckodriver.zip"
Expand-Archive -Path "$toolsDir\geckodriver.zip" -DestinationPath $toolsDir -Force
Remove-Item "$toolsDir\geckodriver.zip"
```

✅ GeckoDriver path:
```
%USERPROFILE%\tools\geckodriver.exe
```

</details>

---

### 🧪 Verify Installation

Run in terminal or PowerShell:
```bash
$HOME/tools/firefox/firefox --version
$HOME/tools/geckodriver --version
```

Or in Windows:
```powershell
& "$env:USERPROFILE\tools\Firefox\firefox.exe" --version
& "$env:USERPROFILE\tools\geckodriver.exe" --version
```

### 🚀🚀🚀 7. Run the full End-to-End Pipeline
🎯**protip:** add `--list_available_markers` to double check the available markers  
🎯**protip:** add `--check_output_exist` to rerun the command; skips fovs that already have output  

linux:
```bash
unhuddle \
  --base_path path/to/dir_containing_individual_fov_folders \
  --output_base_path path/to/unhuddle_output \
  --normalisation_markers CD20 CD68 CD11b CD11c CD8a CD3 CD7 CD45RA CD45RO CD15 CD163 Vimentin CD31 CD14 \
  --nuclear_markers DNA1 DNA2 HistoneH3 \
  --create_nuclear_mask \
  --create_deepcell_mask \
  --deepcell_resolution 10 \
  --geckodriver_path path/to/geckodriver_binary \
  --max_workers 1 \
  --create_adata
```
windows powershell
```powershell
unhuddle `
  --base_path path\to\dir_containing_individual_fov_folders `
  --output_base_path path\to\unhuddle_output `
  --normalisation_markers CD20 CD68 CD11b CD11c CD8a CD3 CD7 CD45RA CD45RO CD15 CD163 Vimentin CD31 CD14 `
  --nuclear_markers DNA1 DNA2 HistoneH3 `
  --create_nuclear_mask `
  --create_deepcell_mask `
  --deepcell_resolution 10 `
  --geckodriver_path path\to\geckodriver_binary `
  --max_workers 1 `
  --create_adata
```

## ⚙️ Pipeline Overview

For each FOV (field of view) folder, the following stages are run:

### 1. **Mask Processing & Interaction Computation**
- Segment cells, nuclei, and membranes.
- Compute:
  - Border interactions (from membrane adjacency)
  - Background interactions (from empty space)
- Merge into a typed interaction dictionary.

### 2. **Feature Extraction**
- Extract per-cell morphological features (area, eccentricity, etc.)
- Exported to CSV.

### 3. **Protein Intensity Extraction**
- Compute marker intensities per segmented cell (excluding nuclear markers).
- Exported to CSV.

### 4. **Object-Intensity Reallocation**
- Reorganize per-pixel interactions into per-cell structure.
- Allocate intensities based on interaction-weighted mean signal.
- *(Graphical schematic to be added here 🔜)*

### 5. **Intensity Settlement**
- Merge morphology and intensity features.
- Apply corrections and output per-measure-type summaries (keeps original available):
  - `/unhuddle_sum/{fov}.csv`, `/original_sum/{fov}.csv`


### 6. **Normalized Intensity Calculation**
- Normalize per-cell values by a weighted mean of top **normalisation markers** (e.g., CD45, CD3, Vimentin).
- Normalisation markers are broadly expressed (typically membrane or cytoplasmic) and support robust within-cell-type comparisons of functional markers like checkpoint inhibitors.
- Values are then robustly scaled per-marker using [0.1, 99.9] percentiles.
- Output: normalized per-cell intensity matrix (values in [0, 1]).
  - `/unhuddle_normalized/{fov}.csv`, `/original_normalized/{fov}.csv`

### 7. **Optional DeepCell Mask Creation**
- Generate RGB overlays from selected markers.
- Use Selenium to upload to DeepCell.org, download results, and integrate mask into downstream analysis.

### 8. **Optional save all the FOV data to Anndata object**
- integrated h5ad object ready for use with scanpy or for rendering annotated FOV images

## 🧰 Available Arguments

### 📂 Input / Output

| Argument                     | Description |
|-----------------------------|-------------|
| `--base_path`               | **Required** Path to folder containing FOV subfolders |
| `--output_base_path`       | **Required** Where to write processed outputs (will be created) |
| `--fovs`                   | List of FOV folder names to process (optional) |
| `--mask_pattern`           | Glob pattern(s) for user imported mask files (e.g. `*_mask_0.tiff`). NB do not use `*.ome.tiff` |
| `--check_output_exist`     | Skip FOVs if output already exists |
| `--create_adata`           | Reconciles all data into a single `.h5ad` file |

### 🎨 Marker definitions

| Argument                     | Description |
|-----------------------------|-------------|  
| `--normalisation_markers` | **Required** unless using `--list_available_markers` |
| `--nuclear-markers`            | **Required** unless using `--list_available_markers` |  
| `--list_available_markers` | List markers and exit |

### ⚙️ Processing Options

| Argument                     | Description |
|-----------------------------|-------------|
| `--max_workers`            | Number of parallel processes (default: 1) |
| `--create_nuclear_mask`    | Generate nuclear mask from cell mask (enables N/C ratio) |
| `--create_deepcell_mask`   | Generate RGB overlay & segment via webloader with DeepCell Mesmer|

### 🌐 DeepCell Settings 

| Argument                   | Description |
|---------------------------|-------------|
| `--geckodriver_path`       | Path to geckodriver for Selenium. **Required when** `--create_deepcell_mask` is used. |
| `--deepcell_url`           | URL of the DeepCell website to connect to (default: `http://www.deepcell.org`). |
| `--deepcell_resolution`    | Objective magnification used for DeepCell overlay creation. Must be one of:<br><br>• `10` → 10x (1 μm/pixel)<br>• `20` → 20x (0.5 μm/pixel)<br>• `40` → 40x (0.25 μm/pixel)<br>• `60` → 60x (0.1667 μm/pixel)<br>• `100` → 100x (0.1 μm/pixel)<br><br>**Required when** `--create_deepcell_mask` is used. |

### 🎨 RGB overlay creation
| Argument                   | Description |
|---------------------------|-------------|
| `--nuclear-markers_overlay`            | Markers for red channel override (nuclear) default: use `nuclear_markers`|  
| `--membrane-markers_overlay`          | Markers for green channel override (membrane/cytoplasm) default: use `normalisation_markers` |  
| `--blue-markers`           | Optional markers for blue channel |

### 🪵 Logging

| Argument                     | Description |
|-----------------------------|-------------|
| `--log-level`              | One of: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---
## 🧬 UNHUDDLE AnnData Object Guide
### 🧬 `adata.obs` — Per-cell annotations

| Column             | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `cell_id` *(index)* | Unique cell identifier: **`{fov}_{Label}`**, e.g., `P23_1_42`               |
| `fov`              | Field of View name: **`{patientID}_{FOVnumber}`**, e.g., `P23_1`             |
| `patient_id`       | Derived from FOV name, e.g., `P23`                                           |
| `summed_intensity` | Total protein intensity for that cell (sum of normalized values)            |
| `QC_no_nucleus`    | Boolean flag indicating no nuclear signal                                   |
| `...`              | All other morphology features (area, eccentricity, etc.)                    |

---

### 🎯 `adata.X` — Normalized protein expression

- Shape: `[n_cells, n_markers]`
- Values are in `[0, 1]`, robustly scaled
- Marker names are in `adata.var_names`

---

### 📚 `adata.var_names`

- List of markers (channels) used for quantification

---

### 🌍 `adata.obsm`

| Key               | Description                                               |
|------------------|-----------------------------------------------------------|
| `spatial`         | `[Centroid_Row, Centroid_Col]`                            |
| `nuclear_spatial` | `[Nucleus_Centroid_Row, Nucleus_Centroid_Col]`           |

---

### 🧬 `adata.layers`

| Layer Key         | Description                                               |
|-------------------|-----------------------------------------------------------|
| `sum_unhuddle`    | Corrected per-cell intensities before normalization       |
| `sum_original`    | Raw intensities prior to interaction reallocation         |

---

### 🧪 `adata.uns`

| Key                | Description                                              |
|--------------------|----------------------------------------------------------|
| `marker-list`      | List of measured marker names                            |
| `fov-list`         | All FOV identifiers (e.g., `P23_1`)                      |
| `patient_id-list`  | All unique patient IDs                                   |
| `spatial`          | Dict of segmentation masks per FOV                       |
| → `spatial[fov]['segmentation']` | 2D numpy array with segmentation labels   |

---

## 🧠 Identifier Format

- `cell_id` = `{fov}_{Label}`  
  Example: `P23_1_42`

- `fov` = `{patientID}_{FOVnumber}`  
  Example: `P23_1`

- `Label` = integer ID in segmentation masks (i.e., pixel regions in `.tiff`)

---



## 📎 Notes
✅ Python Compatibility:
Requires Python 3.8 or higher

🧰 Core Dependencies:

numpy, pandas — numerical and tabular processing

scikit-image, scipy, tifffile — image and mask operations

selenium, requests — automation for DeepCell.org

tqdm — progress bar visualization

🧠 Image Input Expectations:

Marker files should follow the pattern: {marker}.ome.tiff

Data shape: H x W (2D)

Data type: float32 or float64 (from denoising like PENGUIN)

🧩 Segmentation Mask Guidelines:

Auto-detected via --mask_pattern (default: *_0.tiff)

Must be 2D (H x W) or stack format (Z x H x W, Z is squeezed)

If no mask is provided, DeepCell can generate one using --create_deepcell_mask

🔖 Label Convention:

Each cell should have a unique integer label

These labels form the link between segmentation, intensity, and feature CSVs

Optional: match {fov}_{label} for external compatibility

🚀 Parallel Processing:

Use --max_workers to process multiple FOVs concurrently

--check_output_exist skips already-processed FOVs

💡 Normalisation Markers:

Markers like CD3, CD45, Vimentin used for intra-cell normalization

Must be broadly expressed and phenotype-informative

Crucial for deconvolution of functional marker signal

## 📊 Coming Soon
Graphical breakdown of reallocation logic
Support for additional normalization schemes (e.g. quantile)

UNHUDDLE will be installable via `pip install unhuddle`

## 📣 Citation & License
This tool is part of an ongoing research pipeline for high-dimensional tissue profiling originating from the Alizadeh laboratory at Stanford School of Medicine.
Please cite appropriately once a manuscript is available. Open-source license to be defined.


