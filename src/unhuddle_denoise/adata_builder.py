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
    """Reconciles processed FOV outputs into a single AnnData object with robust marker alignment."""

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

    fovs = [Path(p).stem for p in glob.glob(str(Path(dirs["unhuddle_norm"]) / "*.csv"))]
    all_obs = []
    all_X = []
    all_layers = defaultdict(list)
    all_obsm_spatial = []
    denoised_fovs = []
    fov_count = 0

    # Step 1: Explicitly define global marker list from reference CSV
    reference_csv = next(Path(dirs["unhuddle_norm"]).glob("*.csv"))
    reference_df = pd.read_csv(reference_csv)
    markers = [col for col in reference_df.columns if col != "Label"]
    logger.info(f"✅ Using reference markers from {reference_csv.name}: {markers}")

    for fov in tqdm(fovs, desc="Constructing AnnData"):
        p = lambda k: Path(dirs[k]) / f"{fov}.csv" if dirs.get(k) else None
        paths = {
            "intensity": p("unhuddle_norm"),
            "sum": p("unhuddle_sum"),
            "orig_sum": p("original_sum"),
            "morph": p("morph"),
            "denoised_intensity": p("unhuddle_denoised_norm"),
            "denoised_sum": p("unhuddle_denoised_sum"),
            "protein": p("protein")
        }

        if not all(paths[k] and paths[k].is_file() for k in ["intensity", "sum", "orig_sum", "morph"]):
            logger.warning(f"⚠️ Skipping FOV '{fov}': missing required files.")
            continue

        intensity = convert_numeric(load_df(paths["intensity"], fov))
        sum_unhuddle = convert_numeric(load_df(paths["sum"], fov))
        sum_orig = convert_numeric(load_df(paths["orig_sum"], fov))
        morph = convert_numeric(load_df(paths["morph"], fov))
        denoised_intensity = convert_numeric(load_df(paths["denoised_intensity"], fov)) if paths["denoised_intensity"] and paths["denoised_intensity"].is_file() else None
        denoised_sum = convert_numeric(load_df(paths["denoised_sum"], fov)) if paths["denoised_sum"] and paths["denoised_sum"].is_file() else None
        protein_df = convert_numeric(load_df(paths["protein"], fov)) if paths["protein"] and paths["protein"].is_file() else None

        # Check for missing markers explicitly
        missing_markers = set(markers) - set(intensity.columns)
        if missing_markers:
            logger.error(f"⚠️ Missing markers {missing_markers} in FOV '{fov}'; skipping this FOV.")
            continue

        # Explicitly reorder columns to match global marker list
        intensity = intensity[markers]
        sum_unhuddle = sum_unhuddle[markers]
        sum_orig = sum_orig[markers]

        if denoised_intensity is not None:
            denoised_intensity = denoised_intensity[markers]
        if denoised_sum is not None:
            denoised_sum = denoised_sum[markers]

        if all(col in morph.columns for col in ["Nucleus_Area", "Nucleus_Centroid_Row"]):
            morph["QC_no_nucleus"] = morph[["Nucleus_Area", "Nucleus_Centroid_Row"]].isna().any(axis=1)

        morph["fov"] = fov
        morph["patient_id"] = fov.split("_")[0] if "_" in fov else fov

        spatial_coords = morph[["Centroid_Row", "Centroid_Col"]].values if {"Centroid_Row", "Centroid_Col"} <= set(morph.columns) else np.zeros((morph.shape[0], 2))
        all_obsm_spatial.append(spatial_coords)

        morph.drop(columns=["FOV", "Label", "Centroid_Row", "Centroid_Col", "Nucleus_Centroid_Row", "Nucleus_Centroid_Col"], errors="ignore", inplace=True)
        all_obs.append(morph)

        if denoised_intensity is not None and denoised_sum is not None:
            X = denoised_intensity
            all_layers["sum_unhuddle_denoised"].append(denoised_sum.values)
            denoised_fovs.append(fov)
            logger.debug(f"✅ Using denoised intensity data for FOV: {fov}")
            x_src = "normalized_unhuddle_denoised"
        else:
            X = intensity
            logger.debug(f"⚠️ Denoised data missing for FOV: {fov}, falling back to regular normalized intensity")
            x_src = "normalized_unhuddle"

        log_column_stats(X, fov)

        all_X.append(X.values)
        fov_count += 1
        logger.debug(f"📦 [{fov}] → X shape: {X.shape}")

        all_layers["sum_unhuddle"].append(sum_unhuddle.values)
        all_layers["sum_original"].append(sum_orig.values)

        if protein_df is not None:
            exclmem_cols = [col for col in protein_df.columns if col.endswith("_ExclusionMembrane_Sum_Intensity")]
            if exclmem_cols:
                protein_markers = [col.replace("_ExclusionMembrane_Sum_Intensity", "") for col in exclmem_cols]
                if set(protein_markers) == set(markers):
                    # explicitly reorder protein_df columns to match global marker order
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

    logger.debug(f"🧮 Final FOVs used: {fov_count}")
    logger.debug(f"🧪 Denoised FOVs used: {len(denoised_fovs)} → {denoised_fovs}")

    adata = AnnData(
        X=np.vstack(all_X),
        obs=pd.concat(all_obs),
        var=pd.DataFrame(index=markers),
        obsm={"X_spatial": np.vstack(all_obsm_spatial)}
    )

    for key, arrays in all_layers.items():
        if arrays:
            adata.layers[key] = np.vstack(arrays)
            logger.debug(f"📊 Layer '{key}' shape: {adata.layers[key].shape}")

    adata.obs["summed_intensity"] = adata.layers["sum_unhuddle"].sum(axis=1)
    adata.uns["X_source"] = x_src
    adata.uns["fov-list"] = sorted(adata.obs["fov"].unique().tolist())
    adata.uns["patient_id-list"] = sorted(adata.obs["patient_id"].unique().tolist())
    adata.uns["marker-list"] = markers

    adata.write_h5ad(adata_output_path)
    logger.info(f"✅ AnnData saved to: {adata_output_path}")

    return adata

