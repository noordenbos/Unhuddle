import os
import glob
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict, Counter
import concurrent.futures
from tqdm import tqdm
from anndata import AnnData, concat
from tifffile import imread
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*converted to numpy array with dtype.*")

import os
import glob
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
from tqdm import tqdm
from anndata import AnnData
from tifffile import imread
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*converted to numpy array with dtype.*")


def build_adata_from_outputs(dirs: dict, working_path: str, output_adata_name: str = "adata1.h5ad", max_workers: int = 16):
    """Reconciles processed FOV outputs into a single AnnData object."""

    if not os.path.isdir(dirs["adata"]):
        raise FileNotFoundError(f"🛑 Output directory not found: {dirs['adata']}")
    if not os.path.isdir(working_path):
        raise FileNotFoundError(f"🛑 Working path (input FOVs) not found: {working_path}")

    adata_output_path = os.path.join(dirs["adata"], output_adata_name)
    qc_dir = dirs["QC"]

    os.makedirs(os.path.dirname(adata_output_path), exist_ok=True)
    os.makedirs(qc_dir, exist_ok=True)

    logger.info(f"[INFO] Saving output to: {adata_output_path}")
    logger.info(f"Creating QC figures in: {qc_dir}")

    def get_fov_list():
        files = glob.glob(os.path.join(dirs["unhuddle_norm"], "*.csv"))
        return [os.path.splitext(os.path.basename(f))[0] for f in files]

    def load_df(path, fov):
        df = pd.read_csv(path)
        df["Label"] = df["Label"].astype(int)
        df["cell_id"] = f"{fov}_" + df["Label"].astype(str)
        return df

    def convert_numeric(df, exclude=("Label", "cell_id")):
        numeric = df.select_dtypes(include=np.number).columns.difference(exclude)
        df[numeric] = df[numeric].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
        return df

    fovs = get_fov_list()
    all_obs = []
    all_X = []
    all_layers = {
        "sum_unhuddle": [],
        "sum_original": [],
        "sum_unhuddle_denoised": [],
        "ExclMem_Sum": []
    }
    all_obsm_spatial = []
    denoised_fovs = []
    fov_count = 0

    for fov in tqdm(fovs, desc="Constructing AnnData"):
        layer_counts = defaultdict(int)
        paths = {
            "intensity": os.path.join(dirs["unhuddle_norm"], f"{fov}.csv"),
            "sum": os.path.join(dirs["unhuddle_sum"], f"{fov}.csv"),
            "orig_sum": os.path.join(dirs["original_sum"], f"{fov}.csv"),
            "morph": os.path.join(dirs["morph"], f"{fov}.csv"),
            "denoised_intensity": os.path.join(dirs["unhuddle_denoised_norm"], f"{fov}.csv"),
            "denoised_sum": os.path.join(dirs["unhuddle_denoised_sum"], f"{fov}.csv"),
            "protein": os.path.join(dirs["protein"], f"{fov}.csv"),
        }

        if not all(os.path.exists(paths[k]) for k in ["intensity", "sum", "orig_sum", "morph"]):
            continue

        intensity = convert_numeric(load_df(paths["intensity"], fov))
        sum_unhuddle = convert_numeric(load_df(paths["sum"], fov))
        sum_orig = convert_numeric(load_df(paths["orig_sum"], fov))
        morph = convert_numeric(load_df(paths["morph"], fov))
        denoised_intensity = convert_numeric(load_df(paths["denoised_intensity"], fov)) if os.path.exists(paths["denoised_intensity"]) else None
        denoised_sum = convert_numeric(load_df(paths["denoised_sum"], fov)) if os.path.exists(paths["denoised_sum"]) else None
        protein_df = convert_numeric(load_df(paths["protein"], fov)) if os.path.exists(paths["protein"]) else None

        for df in [intensity, sum_unhuddle, sum_orig, morph, denoised_intensity, denoised_sum, protein_df]:
            if df is not None:
                df.set_index("cell_id", inplace=True)

        if all(col in morph.columns for col in ["Nucleus_Area", "Nucleus_Centroid_Row"]):
            morph["QC_no_nucleus"] = morph[["Nucleus_Area", "Nucleus_Centroid_Row"]].isna().any(axis=1)

        morph["fov"] = fov
        morph["patient_id"] = fov.split("_")[0] if "_" in fov else fov

        spatial_coords = morph[["Centroid_Row", "Centroid_Col"]].values if "Centroid_Row" in morph and "Centroid_Col" in morph else None
        all_obsm_spatial.append(spatial_coords)

        morph.drop(columns=["FOV", "Label", "Centroid_Row", "Centroid_Col", "Nucleus_Centroid_Row", "Nucleus_Centroid_Col"], errors="ignore", inplace=True)
        all_obs.append(morph)

        if denoised_intensity is not None and denoised_sum is not None:
            X = denoised_intensity.drop(columns=["Label"], errors="ignore")
            all_layers["sum_unhuddle_denoised"].append(denoised_sum.drop(columns=["Label"], errors="ignore").values)
            X_source = "normalized_unhuddle_denoised"
            denoised_fovs.append(fov)
            logger.debug(f"✅ Using denoised intensity data for FOV: {fov}")
        else:
            X = intensity.drop(columns=["Label"], errors="ignore")
            X_source = "normalized_unhuddle"
            logger.debug(f"⚠️ Denoised data missing for FOV: {fov}, falling back to regular normalized intensity")

        all_X.append(X.values)
        fov_count += 1
        logger.debug(f"📦 [{fov}] → X shape: {X.values.shape}")

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
    adata.uns["X_source"] = "normalized_unhuddle_denoised" if all_layers["sum_unhuddle_denoised"] else "normalized_unhuddle"
    adata.uns["fov-list"] = sorted(adata.obs["fov"].unique().tolist())
    adata.uns["patient_id-list"] = sorted(adata.obs["patient_id"].unique().tolist())
    adata.uns["marker-list"] = list(adata.var_names)

    adata.uns["spatial"] = {}
    for fov in fovs:
        mask_path = os.path.join(working_path, fov, "deepcel_mask.tiff")
        if os.path.exists(mask_path):
            adata.uns["spatial"][fov] = {"segmentation": imread(mask_path)}

    if "dr" in dirs and fovs:
        fitsne_path = os.path.join(dirs["dr"], f"{fovs[0]}.csv")
        if os.path.exists(fitsne_path):
            coords = pd.read_csv(fitsne_path).values
            adata.obsm["X_fitsne"] = coords
            logger.info(f"✅ Loaded DR coordinates for FOV: {fovs[0]}")
        else:
            logger.warning(f"⚠️ DR coordinate file not found: {fitsne_path}")
    else:
        logger.info("ℹ️ Skipping DR coordinate load — 'dr' key not in dirs or no FOVs present.")

    adata.write_h5ad(adata_output_path)
    print(f"AnnData saved to: {adata_output_path}\n\n")
    return adata

