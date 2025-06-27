import warnings
from tifffile import imread
import logging
import os, glob
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
from anndata import AnnData
from pathlib import Path

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*converted to numpy array with dtype.*")
logging.getLogger("anndata").setLevel(logging.WARNING)
def load_df(path: str, fov: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Label"] = df["Label"].astype(int)
    df["cell_id"] = f"{fov}_" + df["Label"].astype(str)
    return df.set_index("cell_id")

def convert_numeric(df: pd.DataFrame, exclude=("Label",)) -> pd.DataFrame:
    cols = df.select_dtypes(include=np.number).columns.difference(exclude)
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

def log_column_stats(X: pd.DataFrame, fov: str):
    for col in X.columns:
        data = X[col].values
        nonzero = np.count_nonzero(data)
        logger.debug(
            f"[{fov}] {col}: min={np.min(data):.3g}, max={np.max(data):.3g}, mean={np.mean(data):.3g}, std={np.std(data):.3g}, nonzero={nonzero}/{len(data)}"
        )

def build_adata_from_outputs(dirs: dict, working_path: str, output_adata_name: str = "adata1.h5ad", max_workers: int = 16):
    """Reconciles processed FOV outputs into a single AnnData object with robust marker alignment and segmentation masks."""

    adata_dir = Path(dirs["adata"])
    if not adata_dir.is_dir():
        raise FileNotFoundError(f"🛑 Output directory not found: {adata_dir}")
    if not Path(working_path).is_dir():
        raise FileNotFoundError(f"🛑 Working path not found: {working_path}")

    adata_output_path = adata_dir / output_adata_name
    qc_dir = Path(dirs["QC"])
    qc_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[INFO] Saving output to: {adata_output_path}")
    logger.info(f"Creating QC figures in: {qc_dir}")

    fovs = [Path(p).stem for p in glob.glob(str(Path(dirs["normalized_unhuddle"]) / "*.csv"))]
    all_obs = []
    all_X = []
    all_layers = defaultdict(list)
    all_obsm_spatial = []
    denoised_fovs = []
    fov_count = 0

    # Step 1: Explicitly define global marker list from reference CSV
    reference_csv = next(Path(dirs["normalized_unhuddle"]).glob("*.csv"))
    reference_df = pd.read_csv(reference_csv)
    markers = [col for col in reference_df.columns if col not in ["Label", "Area"]]
    logger.info(f"✅ Using reference markers from {reference_csv.name}: {markers}")

    for fov in tqdm(fovs, desc="Constructing AnnData"):
        p = lambda k: Path(dirs[k]) / f"{fov}.csv" if dirs.get(k) else None
        paths = {
            "intensity": p("normalized_unhuddle"),
            "sum": p("unhuddle_sum"),
            "orig_sum": p("original_sum"),
            "morph": p("morph"),
            "denoised_intensity": p("normalized_unhuddle_denoised"),
            "denoised_sum": p("unhuddle_denoised_sum"),
            "denoised_intensity_original_style": p("normalized_unhuddle_denoised_original_style"),
            "protein": p("protein"),
            "normalized_unhuddle": p("normalized_unhuddle"),
            "normalized_original": p("normalized_original") if "normalized_original" in dirs else None,
            "normalized_unhuddle_denoised": p("normalized_unhuddle_denoised") if "normalized_unhuddle_denoised" in dirs else None,
            "normalized_unhuddle_denoised_original_style": p("normalized_unhuddle_denoised_original_style") if "normalized_unhuddle_denoised_original_style" in dirs else None,
        }

        if not all(paths[k] and paths[k].is_file() for k in ["intensity", "sum", "orig_sum", "morph"]):
            logger.warning(f"⚠️ Skipping FOV '{fov}': missing required files.")
            continue

        intensity = convert_numeric(load_df(paths["intensity"], fov))[markers]
        sum_unhuddle = convert_numeric(load_df(paths["sum"], fov))[markers]
        sum_orig = convert_numeric(load_df(paths["orig_sum"], fov))[markers]
        morph = convert_numeric(load_df(paths["morph"], fov))
        denoised_intensity = convert_numeric(load_df(paths["denoised_intensity"], fov))[markers] if paths["denoised_intensity"] and paths["denoised_intensity"].is_file() else None
        denoised_sum = convert_numeric(load_df(paths["denoised_sum"], fov))[markers] if paths["denoised_sum"] and paths["denoised_sum"].is_file() else None
        denoised_intensity_original_style = convert_numeric(load_df(paths["denoised_intensity_original_style"], fov))[markers] if paths["denoised_intensity_original_style"] and paths["denoised_intensity_original_style"].is_file() else None
        protein_df = convert_numeric(load_df(paths["protein"], fov)) if paths["protein"] and paths["protein"].is_file() else None

        if all(col in morph.columns for col in ["Nucleus_Area", "Nucleus_Centroid_Row"]):
            morph["QC_no_nucleus"] = morph[["Nucleus_Area", "Nucleus_Centroid_Row"]].isna().any(axis=1)

        morph["fov"] = fov
        morph["patient_id"] = fov.split("_")[0] if "_" in fov else fov

        spatial_coords = morph[["Centroid_Row", "Centroid_Col"]].values if {"Centroid_Row", "Centroid_Col"} <= set(morph.columns) else np.zeros((morph.shape[0], 2))
        all_obsm_spatial.append(spatial_coords)

        morph.drop(columns=["FOV", "Label", "Centroid_Row", "Centroid_Col", "Nucleus_Centroid_Row", "Nucleus_Centroid_Col"], errors="ignore", inplace=True)
        all_obs.append(morph)

        # Always use the regular normalized intensity for X
        X = intensity
        x_src = "normalized_unhuddle"
        logger.debug(f"✅ Using regular normalized intensity data for FOV: {fov}")

        # Add normalized layers for each sum branch if present
        # 1. normalized_unhuddle (from normalized_unhuddle)
        norm_unhuddle_path = paths["normalized_unhuddle"]
        if norm_unhuddle_path is not None and norm_unhuddle_path.is_file():
            norm_unhuddle = convert_numeric(load_df(str(norm_unhuddle_path), fov))[markers]
            all_layers["normalized_unhuddle"].append(norm_unhuddle.values)
        # 2. normalized_original (from normalized_original)
        norm_original_path = paths["normalized_original"]
        if norm_original_path is not None and norm_original_path.is_file():
            norm_original = convert_numeric(load_df(str(norm_original_path), fov))[markers]
            all_layers["normalized_original"].append(norm_original.values)
        # 3. normalized_unhuddle_denoised (from normalized_unhuddle_denoised)
        norm_denoised_path = paths["normalized_unhuddle_denoised"]
        if norm_denoised_path is not None and norm_denoised_path.is_file():
            norm_denoised = convert_numeric(load_df(str(norm_denoised_path), fov))[markers]
            all_layers["normalized_unhuddle_denoised"].append(norm_denoised.values)
        # 4. normalized_unhuddle_denoised_original_style (from normalized_unhuddle_denoised_original_style)
        norm_denoised_orig_path = paths["normalized_unhuddle_denoised_original_style"]
        if norm_denoised_orig_path is not None and norm_denoised_orig_path.is_file():
            norm_denoised_orig = convert_numeric(load_df(str(norm_denoised_orig_path), fov))[markers]
            all_layers["normalized_unhuddle_denoised_original_style"].append(norm_denoised_orig.values)

        # If denoised is requested and present, store in a layer
        if denoised_intensity is not None and denoised_sum is not None:
            if "sum_unhuddle_denoised" not in all_layers:
                all_layers["sum_unhuddle_denoised"] = []
            if "unhuddle_denoised" not in all_layers:
                all_layers["unhuddle_denoised"] = []
            all_layers["sum_unhuddle_denoised"].append(denoised_sum.values)
            all_layers["unhuddle_denoised"].append(denoised_intensity.values)
            logger.debug(f"🧪 Experimental: Denoised intensity available for FOV: {fov} (stored in layers)")
        else:
            logger.debug(f"🟡 No denoised data for FOV: {fov} (no layer stored)")

        # If normalized original-style denoised data is present, store in a layer
        if denoised_intensity_original_style is not None:
            if "unhuddle_denoised_original_style" not in all_layers:
                all_layers["unhuddle_denoised_original_style"] = []
            all_layers["unhuddle_denoised_original_style"].append(denoised_intensity_original_style.values)
            logger.debug(f"🧪 Extra: Normalized original-style denoised intensity available for FOV: {fov} (stored in layers)")
        else:
            logger.debug(f"🟡 No normalized original-style denoised data for FOV: {fov} (no layer stored)")

        # Check for extra original-style CSV files in the separate directory
        if "unhuddle_denoised_sum_original_style" in dirs:
            original_style_path = os.path.join(dirs["unhuddle_denoised_sum_original_style"], f"{fov}.csv")
            if os.path.exists(original_style_path):
                original_style_df = convert_numeric(load_df(original_style_path, fov))[markers]
                if "sum_unhuddle_denoised_original_style" not in all_layers:
                    all_layers["sum_unhuddle_denoised_original_style"] = []
                all_layers["sum_unhuddle_denoised_original_style"].append(original_style_df.values)
                logger.debug(f"🧪 Extra: Original-style denoised sum loaded for FOV: {fov}")

        log_column_stats(X, fov)

        all_X.append(X.values)
        fov_count += 1
        logger.debug(f"📦 [{fov}] → X shape: {X.shape}")

        all_layers["sum_unhuddle"].append(sum_unhuddle.values)
        all_layers["sum_original"].append(sum_orig.values)

        if protein_df is not None:
            exclmem_cols = [col for col in protein_df.columns if col.endswith("_ExclusionMembrane_Sum_Intensity")]
            protein_markers = [col.replace("_ExclusionMembrane_Sum_Intensity", "") for col in exclmem_cols]
            if set(protein_markers) == set(markers):
                ordered_exclmem_cols = [f"{m}_ExclusionMembrane_Sum_Intensity" for m in markers]
                all_layers["ExclMem_Sum"].append(protein_df[ordered_exclmem_cols].values)
                logger.info(f"✅ Reconstructed layer 'ExclMem_Sum' from protein_features for FOV: {fov}")
            else:
                missing_in_protein = set(markers) - set(protein_markers)
                extra_in_protein = set(protein_markers) - set(markers)
                logger.warning(
                    f"⚠️ Protein markers mismatch in FOV '{fov}': "
                    f"missing in protein={missing_in_protein}, extra in protein={extra_in_protein}; "
                    f"skipping ExclMem_Sum layer."
                )

    obs_df = pd.concat(all_obs)
    obs_df["fov"] = obs_df["fov"].astype("category")
    obs_df["patient_id"] = obs_df["patient_id"].astype("category")

    adata = AnnData(
        X=np.vstack(all_X),
        obs=obs_df,
        var=pd.DataFrame(index=markers),
        obsm={"X_spatial": np.vstack(all_obsm_spatial)}
    )


    for key, arrays in all_layers.items():
        if arrays:
            adata.layers[key] = np.vstack(arrays)
            logger.debug(f"📊 Layer '{key}' shape: {adata.layers[key].shape}")

    # ✅ Restored segmentation mask loading with debug logging
    adata.uns["spatial"] = {}
    for fov in fovs:
        mask_path = Path(working_path) / fov / "deepcel_mask.tiff"
        if mask_path.is_file():
            segmentation = imread(mask_path)
            adata.uns["spatial"][fov] = {"segmentation": segmentation}
            logger.debug(f"✅ Loaded segmentation mask for {fov}, shape: {segmentation.shape}")
        else:
            logger.warning(f"⚠️ Mask file not found for {fov}: {mask_path}")

    adata.obs["summed_intensity"] = adata.layers["sum_unhuddle"].sum(axis=1)
    adata.uns["X_source"] = x_src
    adata.uns["fov-list"] = sorted(adata.obs["fov"].unique().tolist())
    adata.uns["patient_id-list"] = sorted(adata.obs["patient_id"].unique().tolist())
    adata.uns["marker-list"] = markers

    adata.write_h5ad(adata_output_path)
    logger.info(f"✅ AnnData saved to: {adata_output_path}")

    return adata

