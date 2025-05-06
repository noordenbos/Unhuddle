import os
from typing import Dict, Optional, Union, Any

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


def infer_apex_from_hexbin(area_filt, intensity_filt, gridsize=80, density_quantile=0.1, min_cells_per_bin=10):
    """Infer apex from hexbin counts."""
    hb = plt.hexbin(area_filt, intensity_filt, gridsize=gridsize, bins='log', cmap='Greys')
    plt.close()

    counts = hb.get_array()
    xbins = hb.get_offsets()[:, 0]
    ybins = hb.get_offsets()[:, 1]

    density_thresh = np.quantile(counts, density_quantile)
    keep_mask = (counts > density_thresh) & (counts >= min_cells_per_bin)

    if not np.any(keep_mask):
        return np.nan, np.nan

    top_x = xbins[keep_mask]
    top_y = ybins[keep_mask]
    peak_idx = np.argmax(top_y)
    apex_area = top_x[peak_idx]
    apex_intensity = top_y[peak_idx]

    return apex_area, apex_intensity


def save_signal_noise_qc_from_df(
    df: pd.DataFrame,
    markers: list[str],
    output_pdf="signal_noise_qc.pdf",
    output_png_dir="signal_noise_qc_pngs",  # Directory for PNGs
    area_col="Area",
    layer_suffix="_ExclusionMembrane_Sum_Intensity",
    apex_anchor_x=15,
    apex_anchor_y=0,
    gridsize=80,
    density_quantile=0.1,
    min_cells_per_bin=10,
    cols=2,  # Two markers per row
    x_max=300
):
    logger = logging.getLogger("unhuddle")
    logger.info("📊 Starting signal/noise QC plotting for %d markers...", len(markers))

    n = len(markers)
    rows = int(np.ceil(n / cols))

    # Create output directory for PNGs
    os.makedirs(output_png_dir, exist_ok=True)

    with PdfPages(output_pdf) as pdf:
        fig, axs = plt.subplots(rows, cols * 2, figsize=(cols * 5, rows * 5), dpi=150)  # Adjusted for residuals
        axs = axs.flatten()

        for i, marker in enumerate(markers):
            logger.debug(f"🔬 Processing marker: {marker}")
            colname = f"{marker}{layer_suffix}"
            if colname not in df.columns or area_col not in df.columns:
                logger.warning(f"⚠️ Missing required columns for marker '{marker}', skipping.")
                axs[i * 2].text(0.5, 0.5, "Data missing", ha='center', va='center', transform=axs[i * 2].transAxes)
                axs[i * 2 + 1].axis("off")  # Hide the residual plot if data is missing
                continue

            area = df[area_col].values
            intensity = df[colname].values

            mask = area >= 10
            area_filt = area[mask]
            intensity_filt = intensity[mask]

            # Hexbin plot
            ax_hexbin = axs[i * 2]
            hb = ax_hexbin.hexbin(area_filt, intensity_filt, gridsize=gridsize, cmap='Greys', bins='log', mincnt=1)

            # Scatter plot (background)
            ax_hexbin.scatter(area_filt, intensity_filt, color='blue', alpha=0.1, s=5)  # Scatter behind hexbin

            counts = hb.get_array()
            xbins = hb.get_offsets()[:, 0]
            ybins = hb.get_offsets()[:, 1]

            density_thresh = np.quantile(counts, density_quantile)
            keep_mask = (counts > density_thresh) & (counts >= min_cells_per_bin)

            if not np.any(keep_mask):
                logger.warning(f"⚠️ No valid apex region for marker '{marker}', skipping fit.")
                ax_hexbin.text(0.5, 0.5, "No valid peak", ha='center', va='center', transform=ax_hexbin.transAxes, color='red')
                axs[i * 2 + 1].axis("off")  # Hide the residual plot if no valid peak
                continue

            top_x = xbins[keep_mask]
            top_y = ybins[keep_mask]
            peak_idx = np.argmax(top_y)
            apex_area = top_x[peak_idx]
            apex_intensity = top_y[peak_idx]

            # Fit the signal using the apex and provided anchor point.
            signal_slope = (apex_intensity - apex_anchor_y) / (apex_area - apex_anchor_x)
            signal_intercept = apex_anchor_y - signal_slope * apex_anchor_x
            x_signal = np.linspace(apex_anchor_x, x_max, 200)
            y_signal = signal_slope * x_signal + signal_intercept
            ax_hexbin.plot(x_signal, y_signal, color='orange', lw=2, label="Signal fit")

            # Infer the noise slope from hexbin bins above the apex.
            noise_mask = (xbins > apex_area) & (counts > density_thresh) & (counts >= min_cells_per_bin)
            if np.any(noise_mask):
                X_noise = xbins[noise_mask].reshape(-1, 1)
                y_noise = ybins[noise_mask]
                X_noise_rel = (X_noise - apex_area)  # relative to the apex area
                model = LinearRegression(fit_intercept=False).fit(X_noise_rel, y_noise)
                noise_slope = model.coef_[0]

                x_noise = np.linspace(apex_area, x_max, 200)
                y_noise = noise_slope * (x_noise - apex_area)
                ax_hexbin.plot(x_noise, y_noise, color='green', lw=2, label="Noise fit")

            # Mark apex
            ax_hexbin.plot(apex_area, apex_intensity, 'ro', label=f"Apex @ {apex_area:.1f}")
            ax_hexbin.axvline(apex_area, linestyle='--', color='orange')
            ax_hexbin.set_title(marker)
            ax_hexbin.set_xlabel("Area")
            ax_hexbin.set_ylabel("Intensity")
            ax_hexbin.legend(fontsize=8)

            # --- Residuals Plot ---
            residuals = intensity_filt - (signal_slope * area_filt + signal_intercept)
            ax_residual = axs[i * 2 + 1]  # Residual plot
            ax_residual.scatter(area_filt, residuals, color='blue', alpha=0.5, s=10)  # Reduced dot size
            ax_residual.axhline(0, color='red', linestyle='--')
            ax_residual.set_title(f"Residuals for {marker}")
            ax_residual.set_xlabel("Area")
            ax_residual.set_ylabel("Residuals")

            # Set y-limits based on residuals
            ax_residual.set_ylim(bottom=min(residuals) * 1.1, top=max(residuals) * 1.1)

            # Use the residuals plot to set the y-limits for the hexbin plot
            ax_hexbin.set_ylim(bottom=min(residuals) * 1.1, top=max(residuals) * 1.1)

            # Set stable x-axis limits
            ax_hexbin.set_xlim(0, x_max)
            ax_residual.set_xlim(0, x_max)

        # Hide any empty axes
        for j in range(len(markers) * 2, len(axs)):
            axs[j].axis("off")

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')

        # Save the figure as PNG
        png_filename = os.path.join(output_png_dir, "signal_noise_qc.png")
        fig.savefig(png_filename, bbox_inches='tight')
        plt.close(fig)

    logger.info("✅ QC PDF saved to: %s", output_pdf)
    logger.info("✅ QC PNG saved to: %s", png_filename)




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
) -> pd.DataFrame:
    area = np.clip(df[area_col].values, min_area, None)
    denoised_df = df.copy()
    metadata = {}

    for marker in markers:
        intensity_col = f"{marker}{layer_suffix}"
        if intensity_col not in df.columns:
            continue

        intensity = df[intensity_col].values.astype(np.float64)
        area_filt = area[area >= min_area]
        intensity_filt = intensity[area >= min_area]

        # Infer apex parameters via a hexbin plot (using log-binning of counts).
        hb = plt.hexbin(area_filt, intensity_filt, gridsize=gridsize, bins='log', cmap='Greys')
        plt.close()
        counts = hb.get_array()
        xbins = hb.get_offsets()[:, 0]
        ybins = hb.get_offsets()[:, 1]

        density_thresh = np.quantile(counts, density_quantile)
        keep_mask = (counts > density_thresh) & (counts >= min_cells_per_bin)
        if not np.any(keep_mask):
            continue

        top_x = xbins[keep_mask]
        top_y = ybins[keep_mask]
        peak_idx = np.argmax(top_y)
        apex_area = top_x[peak_idx]
        apex_intensity = top_y[peak_idx]

        # Fit the signal using the apex and provided anchor point.
        signal_slope = (apex_intensity - signal_anchor_y) / (apex_area - signal_anchor_x)
        signal_intercept = signal_anchor_y - signal_slope * signal_anchor_x
        signal_fit = signal_slope * area + signal_intercept

        # Infer the noise slope from hexbin bins above the apex.
        noise_mask = (xbins > apex_area) & (counts > density_thresh) & (counts >= min_cells_per_bin)
        if np.any(noise_mask):
            X_noise = xbins[noise_mask].reshape(-1, 1)
            y_noise = ybins[noise_mask]
            X_noise_rel = X_noise - apex_area  # relative to the apex area
            noise_model = LinearRegression(fit_intercept=False).fit(X_noise_rel, y_noise)
            noise_slope = noise_model.coef_[0]
        else:
            noise_slope = 0.02  # default fallback value

        noise_fit = np.where(area > apex_area, noise_slope * (area - apex_area), 0)

        # Prepare the model input by stacking the signal and noise fits.
        X = np.stack([signal_fit, noise_fit], axis=1)
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(intensity)
        X_clean = X[valid_mask]
        y_clean = intensity[valid_mask]

        # Fit the full model (signal + noise) to the intensity.
        full_model = LinearRegression().fit(X_clean, y_clean)
        alpha, beta = full_model.coef_
        intercept_model = full_model.intercept_

        signal_contrib = alpha * signal_fit
        noise_contrib = beta * noise_fit

        # Compute residuals after subtracting the model contributions.
        residuals = intensity - (signal_contrib + noise_contrib + intercept_model)

        # Clip any negative residuals to 0.
        residuals_clipped = np.clip(residuals, 0, None)

        # ---- New Masking Logic for Residuals ----
        positive_mask = residuals_clipped > 0

        # Perform area regression only on the positive residuals.
        if np.any(positive_mask):
            X_area = area[positive_mask].reshape(-1, 1)
            y_res = residuals_clipped[positive_mask]
            area_model = LinearRegression().fit(X_area, y_res)
            gamma = area_model.coef_[0]
            intercept_area = area_model.intercept_

            computed_final_denoised = np.zeros_like(residuals_clipped)
            computed_final_denoised[positive_mask] = (
                residuals_clipped[positive_mask] - (gamma * area[positive_mask] + intercept_area)
            )
            computed_final_denoised = np.clip(computed_final_denoised, 0, None)
        else:
            computed_final_denoised = np.zeros_like(residuals_clipped)
            gamma = np.nan
            intercept_area = np.nan

        # Apply robust normalization only on the positive (unmasked) values.
        final_denoised = np.zeros_like(computed_final_denoised)
        positive_norm_mask = computed_final_denoised > 0
        if np.any(positive_norm_mask):
            vmin, vmax = np.percentile(computed_final_denoised[positive_norm_mask], [2, 98])
            if vmax > vmin:
                norm_values = np.clip(
                    (computed_final_denoised[positive_norm_mask] - vmin) / (vmax - vmin), 0, 1
                )
            else:
                norm_values = np.zeros_like(computed_final_denoised[positive_norm_mask])
            final_denoised[positive_norm_mask] = norm_values

        # Save the intermediate (post-clipping) and final denoised intensities.
        denoised_df[f"{marker}_ExclusionMembrane_Denoised_Intensity"] = residuals_clipped
        denoised_df[f"{marker}_ExclusionMembrane_FinalDenoised_Intensity"] = final_denoised

        metadata[marker] = {
            "apex_area": apex_area,
            "apex_intensity": apex_intensity,
            "signal_slope": signal_slope,
            "noise_slope": noise_slope,
            "alpha": alpha,
            "beta": beta,
            "intercept_model": intercept_model,
            "area_regression_coef": gamma,
            "area_regression_intercept": intercept_area
        }

    return {"denoised_df": denoised_df, "metadata": metadata}



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

    result = run_denoising_pipeline_on_dataframe(full_df, markers)
    denoised_df = result["denoised_df"]
    metadata = result["metadata"]
    logger.info(f"Type of denoised_df: {type(denoised_df)}")
    qc_output_pdf = os.path.join(qc_out_dir, "denoiser_QC.pdf")
    qc_output_png = os.path.join(qc_out_dir, "denoiser_QC.png")
    save_signal_noise_qc_from_df(
        df=full_df,
        markers=markers,
        output_pdf=qc_output_pdf,
        output_png_dir=qc_output_png,
        density_quantile=0.1,
        min_cells_per_bin=20
    )
    # Save denoiser metadata
    metadata_path = os.path.join(qc_out_dir, "denoiser_metadata.csv")
    try:
        if isinstance(metadata, pd.DataFrame):
            metadata.to_csv(metadata_path, index=False)
        elif isinstance(metadata, dict):
            metadata_df = pd.DataFrame.from_dict(metadata, orient='index')
            metadata_df.to_csv(metadata_path, index=True)
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

