import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore", message=".*partition.*MaskedArray.*")

logger = logging.getLogger(__name__)



def infer_apex_from_smoothed_histogram(area_filt, intensity_filt, bins=60, sigma=1.2, target_percentile=0.5):
    """Infer apex from smoothed 2D histogram at a fixed percentile (e.g., 0.5%)."""
    hist, xedges, yedges = np.histogram2d(area_filt, intensity_filt, bins=[bins, bins])
    smoothed = gaussian_filter(hist, sigma=sigma)

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(xcenters, ycenters, indexing="ij")

    sorted_vals = np.sort(smoothed.ravel())[::-1]
    cumsum = np.cumsum(sorted_vals)
    cumsum /= cumsum[-1]
    idx_level = np.searchsorted(cumsum, (100 - target_percentile) / 100.)
    level = sorted_vals[min(idx_level, len(sorted_vals) - 1)]

    # Find contours manually
    contour_set = plt.contour(X, Y, smoothed, levels=[level])
    plt.close()

    all_segments = contour_set.allsegs[0]
    if not all_segments:
        return np.nan, np.nan

    # Merge all segments together
    all_points = np.vstack(all_segments)
    apex_idx = np.argmax(all_points[:,1])  # maximize Intensity
    apex_area = all_points[apex_idx, 0]
    apex_intensity = all_points[apex_idx, 1]

    return apex_area, apex_intensity

def save_signal_noise_qc_from_df(
    df: pd.DataFrame,
    markers: list[str],
    output_pdf="signal_noise_qc.pdf",
    area_col="Area",
    layer_suffix="_ExclusionMembrane_Sum_Intensity",
    apex_anchor_x=15,
    apex_anchor_y=0,
    gridsize=80,
    density_quantile=0.1,
    min_cells_per_bin=10,
    cols=3,
    x_max=300
):
    logger = logging.getLogger(__name__)
    logger.info("📊 Starting signal/noise QC plotting for %d markers...", len(markers))

    n = len(markers)
    rows = int(np.ceil(n / cols))

    with PdfPages(output_pdf) as pdf:
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), dpi=150)
        axs = axs.flatten()

        for i, marker in enumerate(markers):
            logger.debug(f"🔬 Processing marker: {marker}")
            colname = f"{marker}{layer_suffix}"
            if colname not in df.columns or area_col not in df.columns:
                logger.warning(f"⚠️ Missing required columns for marker '{marker}', skipping.")
                axs[i].text(0.5, 0.5, "Data missing", ha='center', va='center', transform=axs[i].transAxes)
                continue

            area = df[area_col].values
            intensity = df[colname].values

            mask = area >= 10
            area_filt = area[mask]
            intensity_filt = intensity[mask]

            ax = axs[i]
            # Just for visualization: scatter plot
            ax.hexbin(area_filt, intensity_filt, gridsize=gridsize, cmap='Greys', bins='log', mincnt=1)

            # --- Smoothed histogram contour overlay ---
            hist, xedges, yedges = np.histogram2d(area_filt, intensity_filt, bins=[60, 60])
            smoothed = gaussian_filter(hist, sigma=1.2)
            xcenters = (xedges[:-1] + xedges[1:]) / 2
            ycenters = (yedges[:-1] + yedges[1:]) / 2
            X, Y = np.meshgrid(xcenters, ycenters, indexing="ij")

            sorted_vals = np.sort(smoothed.ravel())[::-1]
            cumsum = np.cumsum(sorted_vals)
            cumsum /= cumsum[-1]

            percentiles = [10, 1, 0.5]
            levels = []
            for p in percentiles:
                idx = np.searchsorted(cumsum, (100 - p) / 100.)
                levels.append(sorted_vals[min(idx, len(sorted_vals) - 1)])
            levels, percentiles = zip(*sorted(zip(levels, percentiles)))  # ensure increasing

            try:
                contour = ax.contour(X, Y, smoothed, levels=levels, colors=['red', 'blue', 'green'], linewidths=1.2)
                fmt = {l: f"{p:.1f}%" for l, p in zip(contour.levels, percentiles)}
                ax.clabel(contour, inline=True, fontsize=7, fmt=fmt)
            except Exception as e:
                logger.warning(f"⚠️ Contour plot failed for {marker}: {e}")
                ax.text(0.5, 0.5, "Contour Error", ha='center', va='center', transform=ax.transAxes)

            # --- Infer apex ---
            apex_area, apex_intensity = infer_apex_from_smoothed_histogram(
                area_filt, intensity_filt, bins=60, sigma=1.2, target_percentile=0.5
            )
            if np.isnan(apex_area):
                ax.text(0.5, 0.5, "No Apex", ha='center', va='center', transform=ax.transAxes)
                continue

            # --- Signal fit ---
            signal_slope = (apex_intensity - apex_anchor_y) / (apex_area - apex_anchor_x)
            signal_intercept = apex_anchor_y - signal_slope * apex_anchor_x
            x_signal = np.linspace(apex_anchor_x, x_max, 200)
            y_signal = signal_slope * x_signal + signal_intercept
            ax.plot(x_signal, y_signal, color='orange', lw=2, label="Signal fit")

            # --- Noise fit using numpy bins ---
            hist, xedges, yedges = np.histogram2d(area_filt, intensity_filt, bins=[gridsize, gridsize])
            xcenters = (xedges[:-1] + xedges[1:]) / 2
            ycenters = (yedges[:-1] + yedges[1:]) / 2
            counts = hist.flatten()
            xbins = np.repeat(xcenters, gridsize)
            ybins = np.tile(ycenters, gridsize)
            density_thresh = np.quantile(counts, density_quantile)

            noise_mask = (xbins > apex_area) & (counts > density_thresh) & (counts >= min_cells_per_bin)
            if np.any(noise_mask):
                X_noise = (xbins[noise_mask] - apex_area).reshape(-1, 1)
                y_noise = ybins[noise_mask]
                noise_model = LinearRegression(fit_intercept=False).fit(X_noise, y_noise)
                noise_slope = noise_model.coef_[0]

                x_noise = np.linspace(apex_area, x_max, 200)
                y_noise = noise_slope * (x_noise - apex_area)
                ax.plot(x_noise, y_noise, color='green', lw=2, label="Noise fit")

            # --- Mark apex ---
            ax.plot(apex_area, apex_intensity, 'o', color='lime', markersize=6, label=f"Apex @ {apex_area:.1f}")
            ax.axvline(apex_area, linestyle='--', color='orange')
            ax.set_title(marker)
            ax.set_xlabel("Area")
            ax.set_ylabel("Intensity")
            ax.legend(fontsize=7)

        # Hide any empty axes
        for j in range(len(markers), len(axs)):
            axs[j].axis("off")

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    logger.info("✅ QC PDF saved to: %s", output_pdf)



import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import logging
from scipy.ndimage import gaussian_filter


def run_denoising_pipeline_on_dataframe(
    df: pd.DataFrame,
    markers: list[str],
    area_col: str = "Area",
    layer_suffix: str = "_ExclusionMembrane_Sum_Intensity",
    signal_anchor_x: float = 15,
    signal_anchor_y: float = 0,
    gridsize: int = 100,
    density_quantile: float = 0.1,
    min_cells_per_bin: int = 10,
    min_area: float = 15,
    store_metadata: bool = True,
) -> pd.DataFrame:
    """
    Adds denoised values for each marker into a DataFrame based on signal/noise cone fitting.
    Area values < `min_area` are clipped instead of excluded.

    Returns:
        DataFrame with additional columns for denoised intensity values and optionally fitting metadata.
    """
    logger = logging.getLogger(__name__)
    area = np.clip(df[area_col].values, min_area, None)
    denoised_df = df.copy()
    metadata = {}

    for marker in markers:
        intensity_col = f"{marker}{layer_suffix}"
        if intensity_col not in df.columns:
            logger.warning(f"⚠️ Marker {marker} missing in dataframe, skipping.")
            continue

        intensity = df[intensity_col].values.astype(np.float64)
        area_filt = area[area >= min_area]
        intensity_filt = intensity[area >= min_area]

        # Infer apex
        apex_area, apex_intensity = infer_apex_from_smoothed_histogram(
            area_filt, intensity_filt, bins=60, sigma=1.2, target_percentile=0.5
        )

        if np.isnan(apex_area) or np.isnan(apex_intensity):
            logger.warning(f"⚠️ Apex inference failed for marker {marker}, skipping.")
            continue

        # --- Signal fit ---
        signal_slope = (apex_intensity - signal_anchor_y) / (apex_area - signal_anchor_x)
        signal_intercept = signal_anchor_y - signal_slope * signal_anchor_x
        signal_fit = signal_slope * area + signal_intercept

        # --- Noise fit (new: pure numpy binning) ---
        hist, xedges, yedges = np.histogram2d(area_filt, intensity_filt, bins=[gridsize, gridsize])
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        counts = hist.flatten()
        xbins = np.repeat(xcenters, gridsize)
        ybins = np.tile(ycenters, gridsize)
        density_thresh = np.quantile(counts, density_quantile)

        noise_mask = (xbins > apex_area) & (counts > density_thresh) & (counts >= min_cells_per_bin)
        if np.any(noise_mask):
            X_noise = (xbins[noise_mask] - apex_area).reshape(-1, 1)
            y_noise = ybins[noise_mask]
            noise_model = LinearRegression(fit_intercept=False).fit(X_noise, y_noise)
            noise_slope = noise_model.coef_[0]
        else:
            noise_slope = 0.02  # fallback

        noise_fit = np.where(area > apex_area, noise_slope * (area - apex_area), 0)

        # --- Full signal + noise model ---
        X = np.stack([signal_fit, noise_fit], axis=1)
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(intensity)
        X_clean = X[valid_mask]
        y_clean = intensity[valid_mask]

        full_model = LinearRegression().fit(X_clean, y_clean)
        alpha, beta = full_model.coef_
        intercept_model = full_model.intercept_

        signal_contrib = alpha * signal_fit
        noise_contrib = beta * noise_fit
        model_pred = signal_contrib + noise_contrib + intercept_model
        residuals = intensity - model_pred

        # --- Residual Clipping and Masking ---
        residuals_clipped = np.clip(residuals, 0, None)
        positive_mask = residuals_clipped > 0

        if np.any(positive_mask):
            X_area = area[positive_mask].reshape(-1, 1)
            y_res = residuals_clipped[positive_mask]
            area_model = LinearRegression().fit(X_area, y_res)
            gamma = area_model.coef_[0]
            intercept_area = area_model.intercept_

            bias_correction = gamma * area + intercept_area
            corrected_residuals = residuals_clipped - bias_correction
            corrected_residuals = np.clip(corrected_residuals, 0, None)
        else:
            corrected_residuals = np.zeros_like(residuals_clipped)
            gamma = np.nan
            intercept_area = np.nan

        # --- Robust normalization ---
        final_denoised = np.zeros_like(corrected_residuals)
        pos_norm_mask = corrected_residuals > 0
        if np.any(pos_norm_mask):
            vmin, vmax = np.percentile(corrected_residuals[pos_norm_mask], [2, 98])
            if vmax > vmin:
                norm_vals = (corrected_residuals[pos_norm_mask] - vmin) / (vmax - vmin)
                norm_vals = np.clip(norm_vals, 0, 1)
            else:
                norm_vals = np.zeros_like(corrected_residuals[pos_norm_mask])
            final_denoised[pos_norm_mask] = norm_vals

        # --- Save outputs ---
        denoised_df[f"{marker}_ExclusionMembrane_Denoised_Intensity"] = residuals_clipped
        denoised_df[f"{marker}_ExclusionMembrane_FinalDenoised_Intensity"] = final_denoised

        if store_metadata:
            metadata[marker] = {
                "apex_area": apex_area,
                "apex_intensity": apex_intensity,
                "signal_slope": signal_slope,
                "noise_slope": noise_slope,
                "alpha": alpha,
                "beta": beta,
                "intercept_model": intercept_model,
                "area_regression_coef": gamma,
                "area_regression_intercept": intercept_area,
                "density_quantile_used": density_quantile,
            }

    return (denoised_df, metadata)



def compute_denoised_reallocation_factors(protein_csv_paths, dirs):
    logger = logging.getLogger(__name__)

    morph_dir = dirs["morph"]
    protein_dir = dirs["protein"]
    qc_out_dir = dirs.get("QC_metadata_denoised", os.path.join(dirs["QC"], "metadata_denoise"))
    os.makedirs(qc_out_dir, exist_ok=True)

    # Derive corresponding morphology paths
    morph_csv_paths = [
        os.path.join(morph_dir, os.path.basename(p))
        for p in protein_csv_paths
    ]

    all_fovs_data = []
    fov_ids = []

    for morph_path, protein_path in zip(morph_csv_paths, protein_csv_paths):
        try:
            morph_df = pd.read_csv(morph_path)
            protein_df = pd.read_csv(protein_path)

            if "Area" not in morph_df.columns:
                raise ValueError(f"'Area' column missing in {morph_path}")

            fov_name = os.path.splitext(os.path.basename(protein_path))[0]
            joint_df = morph_df[["Area"]].join(protein_df, how="inner")
            joint_df["fov"] = fov_name

            all_fovs_data.append(joint_df)
            fov_ids.append(fov_name)

        except Exception as e:
            logger.warning(f"⚠️ Skipping {protein_path}: {e}")

    if not all_fovs_data:
        raise RuntimeError("❌ No valid FOVs found to compute denoised reallocation factors.")

    full_df = pd.concat(all_fovs_data, axis=0, ignore_index=True)

    intensity_cols = [col for col in full_df.columns if col.endswith("_ExclusionMembrane_Sum_Intensity")]
    markers = [col.replace("_ExclusionMembrane_Sum_Intensity", "") for col in intensity_cols]

    logger.info("🚀 Running cohort-wide denoising on %d markers across %d FOVs...", len(markers), len(fov_ids))

    denoised_df, metadata = run_denoising_pipeline_on_dataframe(full_df, markers)

    qc_output_pdf = os.path.join(qc_out_dir, "denoiser_QC.pdf")
    save_signal_noise_qc_from_df(
        df=full_df,
        markers=markers,
        output_pdf=qc_output_pdf,
        density_quantile=0.1,
        min_cells_per_bin=20
    )
    # Save denoiser metadata
    metadata_path = os.path.join(qc_out_dir, "denoiser_metadata.csv")
    try:
        if isinstance(metadata, pd.DataFrame):
            metadata.to_csv(metadata_path, index=False)
        elif isinstance(metadata, dict):
            pd.DataFrame.from_dict(metadata).to_csv(metadata_path, index=False)
        else:
            raise TypeError("Unsupported metadata format; expected DataFrame or dict.")
        logger.info(f"🧾 Saved denoiser metadata: {metadata_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save denoiser metadata: {e}")

    for fov_name, group in denoised_df.groupby("fov"):
        denoised_cols = [
            col for col in group.columns
            if col.endswith("_ExclusionMembrane_Denoised_Intensity") or
               col.endswith("_ExclusionMembrane_FinalDenoised_Intensity")
        ]
        denoised_block = group[denoised_cols].reset_index(drop=True)

        protein_csv_path = os.path.join(protein_dir, f"{fov_name}.csv")
        if not os.path.exists(protein_csv_path):
            logger.warning(f"⚠️ Protein CSV not found for FOV '{fov_name}', skipping update.")
            continue

        try:
            protein_df = pd.read_csv(protein_csv_path)
            cols_to_drop = [col for col in protein_df.columns if col in denoised_block.columns]
            if cols_to_drop:
                logger.debug(f"🧹 Overwriting existing denoised columns for FOV '{fov_name}': {cols_to_drop}")
                protein_df.drop(columns=cols_to_drop, inplace=True)

            updated_df = pd.concat([protein_df.reset_index(drop=True), denoised_block], axis=1)
            updated_df.to_csv(protein_csv_path, index=False)

            logger.info(f"📝 Denoised values updated in protein CSV for FOV: {fov_name}")

        except Exception as e:
            logger.error(f"❌ Failed to update {protein_csv_path}: {e}")

