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
    """Reconciles processed FOV outputs into a single AnnData object."""

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
            continue

        intensity = convert_numeric(load_df(paths["intensity"], fov))
        sum_unhuddle = convert_numeric(load_df(paths["sum"], fov))
        sum_orig = convert_numeric(load_df(paths["orig_sum"], fov))
        morph = convert_numeric(load_df(paths["morph"], fov))
        denoised_intensity = convert_numeric(load_df(paths["denoised_intensity"], fov)) if paths["denoised_intensity"] and paths["denoised_intensity"].is_file() else None
        denoised_sum = convert_numeric(load_df(paths["denoised_sum"], fov)) if paths["denoised_sum"] and paths["denoised_sum"].is_file() else None
        protein_df = convert_numeric(load_df(paths["protein"], fov)) if paths["protein"] and paths["protein"].is_file() else None

        if all(col in morph.columns for col in ["Nucleus_Area", "Nucleus_Centroid_Row"]):
            morph["QC_no_nucleus"] = morph[["Nucleus_Area", "Nucleus_Centroid_Row"]].isna().any(axis=1)

        morph["fov"] = fov
        morph["patient_id"] = fov.split("_")[0] if "_" in fov else fov

        spatial_coords = morph[["Centroid_Row", "Centroid_Col"]].values if {"Centroid_Row", "Centroid_Col"} <= set(morph.columns) else np.zeros((morph.shape[0], 2))
        all_obsm_spatial.append(spatial_coords)

        morph.drop(columns=["FOV", "Label", "Centroid_Row", "Centroid_Col", "Nucleus_Centroid_Row", "Nucleus_Centroid_Col"], errors="ignore", inplace=True)
        all_obs.append(morph)

        if denoised_intensity is not None and denoised_sum is not None:
            X = denoised_intensity.drop(columns=["Label"], errors="ignore")
            all_layers["sum_unhuddle_denoised"].append(denoised_sum.drop(columns=["Label"], errors="ignore").values)
            denoised_fovs.append(fov)
            logger.debug(f"✅ Using denoised intensity data for FOV: {fov}")
            x_src = "normalized_unhuddle_denoised"
        else:
            X = intensity.drop(columns=["Label"], errors="ignore")
            logger.debug(f"⚠️ Denoised data missing for FOV: {fov}, falling back to regular normalized intensity")
            x_src = "normalized_unhuddle"

        log_column_stats(X, fov)

        all_X.append(X.values)
        fov_count += 1
        logger.debug(f"📦 [{fov}] → X shape: {X.shape}")

        all_layers["sum_unhuddle"].append(sum_unhuddle.drop(columns=["Label"], errors="ignore").values)
        all_layers["sum_original"].append(sum_orig.drop(columns=["Label"], errors="ignore").values)

        if protein_df is not None:
            exclmem_cols = [col for col in protein_df.columns if col.endswith("_ExclusionMembrane_Sum_Intensity")]
            if exclmem_cols:
                markers = [col.replace("_ExclusionMembrane_Sum_Intensity", "") for col in exclmem_cols]
                all_layers["ExclMem_Sum"].append(protein_df[exclmem_cols].values)
                var_names = markers
                logger.info(f"✅ Reconstructed layer 'ExclMem_Sum' from protein_features for FOV: {fov}")
            else:
                logger.warning(f"⚠️ No ExclusionMembrane_Sum_Intensity columns found in protein_features for FOV: {fov}")

    logger.debug(f"🧮 Final FOVs used: {fov_count}")
    logger.debug(f"🧪 Denoised FOVs used: {len(denoised_fovs)} → {denoised_fovs}")

    n_obs = sum(x.shape[0] for x in all_X)
    for key, arrs in all_layers.items():
        if arrs:
            shape_sum = sum(x.shape[0] for x in arrs)
            logger.debug(f"📊 Layer '{key}': {shape_sum} rows across {len(arrs)} chunks")
            if key == "sum_unhuddle_denoised":
                assert shape_sum == n_obs, f"❌ Mismatch for layer '{key}': expected {n_obs}, got {shape_sum}"

    var_names = X.columns.tolist() if hasattr(X, "columns") else [f"marker_{i}" for i in range(X.shape[1])]

    adata = AnnData(
        X=np.vstack(all_X),
        obs=pd.concat(all_obs),
        var=pd.DataFrame(index=var_names),
        obsm={"X_spatial": np.vstack(all_obsm_spatial)}
    )
    for key, arrays in all_layers.items():
        if arrays:
            adata.layers[key] = np.vstack(arrays)

    adata.obs["summed_intensity"] = adata.layers["sum_unhuddle"].sum(axis=1)
    adata.uns["X_source"] = x_src
    adata.uns["fov-list"] = sorted(adata.obs["fov"].unique().tolist())
    adata.uns["patient_id-list"] = sorted(adata.obs["patient_id"].unique().tolist())
    adata.uns["marker-list"] = var_names

    adata.uns["spatial"] = {}
    for fov in fovs:
        mask_path = Path(working_path) / fov / "deepcel_mask.tiff"
        if mask_path.is_file():
            adata.uns["spatial"][fov] = {"segmentation": imread(mask_path)}

    adata.write_h5ad(adata_output_path)
    print(f"AnnData saved to: {adata_output_path}\n\n")
    return adata
