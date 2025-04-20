import os
import glob
import numpy as np
import pandas as pd
import warnings
import concurrent.futures
from tqdm import tqdm
from anndata import AnnData, concat
from tifffile import imread
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*converted to numpy array with dtype.*")

def build_adata_from_outputs(output_base_path, working_path=None, output_adata_name="adata1.h5ad", max_workers=16):
    """Reconciles processed FOV outputs into a single AnnData object."""
    if working_path is None:
        raise ValueError("🛑 'working_path' must be explicitly provided — segmentation masks are required.")

    if not os.path.isdir(output_base_path):
        raise FileNotFoundError(f"🛑 Output directory not found: {output_base_path}")

    if not os.path.isdir(working_path):
        raise FileNotFoundError(f"🛑 Working path (input FOVs) not found: {working_path}")

    adata_output_path = os.path.join(output_base_path, "adata_objects", output_adata_name)
    qc_dir = os.path.join(output_base_path, "QC")

    os.makedirs(os.path.dirname(adata_output_path), exist_ok=True)
    os.makedirs(qc_dir, exist_ok=True)

    logger.info(f"[INFO] Saving output to: {adata_output_path}")
    logger.info(f"Creating QC figures in: {qc_dir}")

    def get_fov_list():
        files = glob.glob(f"{output_base_path}/unhuddle_normalized/*.csv")
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

    for fov in tqdm(fovs, desc="Constructing AnnData"):
        paths = {
            "intensity": f"{output_base_path}/unhuddle_normalized/{fov}.csv",
            "sum": f"{output_base_path}/unhuddle_sum/{fov}.csv",
            "orig_sum": f"{output_base_path}/original_sum/{fov}.csv",
            "morph": f"{output_base_path}/morphology_features/{fov}.csv",
            "denoised_intensity": f"{output_base_path}/unhuddle_denoised_normalized/{fov}.csv",
            "denoised_sum": f"{output_base_path}/unhuddle_denoised_sum/{fov}.csv",
            "protein": f"{output_base_path}/protein_features/{fov}.csv",
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

        if denoised_intensity is not None:
            X = denoised_intensity.drop(columns=["Label"], errors="ignore")
            X_source = "normalized_unhuddle_denoised"
        else:
            X = intensity.drop(columns=["Label"], errors="ignore")
            X_source = "normalized_unhuddle"

        all_X.append(X.values)

        all_layers["sum_unhuddle"].append(sum_unhuddle.drop(columns=["Label"], errors="ignore").values)
        all_layers["sum_original"].append(sum_orig.drop(columns=["Label"], errors="ignore").values)
        if denoised_sum is not None:
            all_layers["sum_unhuddle_denoised"].append(denoised_sum.drop(columns=["Label"], errors="ignore").values)

        if protein_df is not None:
            exclmem_cols = [col for col in protein_df.columns if col.endswith("_ExclusionMembrane_Sum_Intensity")]
            if exclmem_cols:
                markers = [col.replace("_ExclusionMembrane_Sum_Intensity", "") for col in exclmem_cols]
                all_layers["ExclMem_Sum"].append(protein_df[exclmem_cols].values)
                var_names = markers
                logger.info(f"✅ Reconstructed layer 'ExclMem_Sum' from protein_features for FOV: {fov}")
            else:
                logger.warning(f"⚠️ No ExclusionMembrane_Sum_Intensity columns found in protein_features for FOV: {fov}")

    adata = AnnData(
        X=np.vstack(all_X),
        obs=pd.concat(all_obs),
        var=pd.DataFrame(index=var_names),
        obsm={"spatial": np.vstack(all_obsm_spatial)}
    )
    for key, arrays in all_layers.items():
        if arrays:
            adata.layers[key] = np.vstack(arrays)

    adata.obs["summed_intensity"] = adata.layers["sum_unhuddle"].sum(axis=1)
    adata.uns["X_source"] = X_source
    adata.uns["fov-list"] = sorted(adata.obs["fov"].unique().tolist())
    adata.uns["patient_id-list"] = sorted(adata.obs["patient_id"].unique().tolist())
    adata.uns["marker-list"] = list(adata.var_names)

    adata.uns["spatial"] = {}
    for fov in fovs:
        mask_path = os.path.join(working_path, fov, "deepcel_mask.tiff")
        if os.path.exists(mask_path):
            adata.uns["spatial"][fov] = {"segmentation": imread(mask_path)}

    fitsne_path = os.path.join(output_base_path, "fitsne_coords", f"{fovs[0]}.csv")
    if os.path.exists(fitsne_path):
        coords = pd.read_csv(fitsne_path).values
        adata.obsm["X_fitsne"] = coords

    adata.write_h5ad(adata_output_path)
    print(f"AnnData saved to: {adata_output_path}\n\n")
    return adata
