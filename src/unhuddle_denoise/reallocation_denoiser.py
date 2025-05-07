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


import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression

def save_signal_noise_qc_from_df(
    df: pd.DataFrame,
    markers: list[str],
    metadata: dict[str, dict],
    output_pdf: str = "signal_noise_qc.pdf",
    output_png_dir: str = "signal_noise_qc_pngs",
    morph_col: str = "Area",
    layer_suffix: str = "_ExclusionMembrane_Sum_Intensity",
    anchor_x: float = 15,
    anchor_y: float = 0,
    cols: int = 2,
) -> None:
    """
    Generate QC plots for noise-based regression and residuals per marker,
    using precomputed metadata. The residual scatter determines axis limits.
    """
    logger = logging.getLogger("unhuddle")
    logger.info("📊 Starting noise QC plotting for %d markers...", len(markers))

    os.makedirs(output_png_dir, exist_ok=True)
    n_rows = int(np.ceil(len(markers) / cols))

    # —— COMBINED PDF ——
    with PdfPages(output_pdf) as pdf:
        fig, axs = plt.subplots(n_rows, cols * 2,
                                figsize=(cols * 5, n_rows * 5),
                                dpi=150)
        axs = axs.flatten()

        for idx, marker in enumerate(markers):
            fit_ax   = axs[2 * idx]
            resid_ax = axs[2 * idx + 1]
            colname  = f"{marker}{layer_suffix}"
            meta     = metadata.get(marker, {})

            # Validate inputs
            if colname not in df.columns or morph_col not in df.columns or not meta:
                fit_ax.text(0.5, 0.5, "No data", ha='center', va='center',
                            transform=fit_ax.transAxes)
                resid_ax.axis('off')
                continue

            area    = df[morph_col].values
            inten   = df[colname].values
            denoise = df[f"{marker}_ExclusionMembrane_Sum_Intensity"].values

            # Fit plot: scatter raw
            mask    = area >= anchor_x
            area_f  = area[mask]; inten_f = inten[mask]
            fit_ax.scatter(area_f, inten_f, color='blue', alpha=0.3, s=5)

            # Plot noise slope line
            xs = np.linspace(area.min(), area.max(), 200)
            ys = meta["noise_slope"] * (xs - anchor_x) + anchor_y
            fit_ax.plot(xs, ys, color='green', lw=2, label="Noise fit")

            # Plot intensity cutoff
            x_cut = meta["morph_cutoff"]
            fit_ax.axvline(x_cut, linestyle='--', color='red',
                           label="intensity cutoff")

            # Optional area-correction
            gamma = meta.get("area_reg_coef", np.nan)
            intercept_area = meta.get("area_reg_intercept", np.nan)
            if not np.isnan(gamma):
                ya = gamma * xs + intercept_area
                fit_ax.plot(xs, ya, color='purple', lw=1, linestyle=':',
                            label="Area correction")

            fit_ax.set_title(marker)
            fit_ax.set_xlabel(morph_col)
            fit_ax.set_ylabel("Intensity")
            fit_ax.legend(fontsize=6)

            # Residual plot
            resid_ax.scatter(area, denoise, color='blue', alpha=0.3, s=5)
            resid_ax.axhline(0, color='red', linestyle='--')

            # Determine axis limits from residuals
            x_min, x_max = area.min(), area.max()
            y_lo, y_hi   = np.percentile(denoise, [1, 99])
            if np.isclose(y_lo, y_hi):
                y_lo -= 0.5; y_hi += 0.5

            resid_ax.set_xlim(x_min, x_max)
            resid_ax.set_ylim(y_lo * 1.1, y_hi * 1.1)

            # Mirror limits on fit plot
            fit_ax.set_xlim(x_min, x_max)
            fit_ax.set_ylim(y_lo * 1.1, y_hi * 1.1)

        # Hide extra axes
        total = 2 * len(markers)
        for j in range(total, len(axs)):
            axs[j].axis('off')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    logger.info("✅ QC PDF saved to %s", output_pdf)

    # —— INDIVIDUAL PNGs ——
    for marker in markers:
        fig, (fit_ax, resid_ax) = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
        colname = f"{marker}{layer_suffix}"
        meta    = metadata.get(marker, {})

        if colname not in df.columns or morph_col not in df.columns or not meta:
            fit_ax.text(0.5, 0.5, "No data", ha='center', va='center',
                        transform=fit_ax.transAxes)
            resid_ax.axis('off')
        else:
            area    = df[morph_col].values
            inten   = df[colname].values
            denoise = df[f"{marker}_ExclusionMembrane_Sum_Intensity"].values

            # Fit plot
            mask    = area >= anchor_x
            area_f  = area[mask]; inten_f = inten[mask]
            fit_ax.scatter(area_f, inten_f, color='blue', alpha=0.3, s=5)

            xs = np.linspace(area.min(), area.max(), 200)
            ys = meta["noise_slope"] * (xs - anchor_x) + anchor_y
            fit_ax.plot(xs, ys, color='green', lw=2)
            fit_ax.axvline(meta["morph_cutoff"], linestyle='--', color='red')

            gamma = meta.get("area_reg_coef", np.nan)
            intercept_area = meta.get("area_reg_intercept", np.nan)
            if not np.isnan(gamma):
                ya = gamma * xs + intercept_area
                fit_ax.plot(xs, ya, color='purple', lw=1, linestyle=':')

            fit_ax.set_title(marker)
            fit_ax.set_xlabel(morph_col)
            fit_ax.set_ylabel("Intensity")

            # Residual plot
            resid_ax.scatter(area, denoise, color='blue', alpha=0.3, s=5)
            resid_ax.axhline(0, color='red', linestyle='--')

            # Determine axis limits from residuals
            x_min, x_max = area.min(), area.max()
            y_lo, y_hi = np.percentile(denoise, [1, 99])

            # Explicit padding (5% of the data range)
            padding_x = 0.05 * (x_max - x_min)
            padding_y = 0.05 * (y_hi - y_lo)

            # Avoid collapse if the range is too narrow
            if np.isclose(y_lo, y_hi) or (y_hi - y_lo) < 1e-2:
                y_lo -= 0.5
                y_hi += 0.5

            # Apply padding explicitly
            x_min_padded = x_min - padding_x
            x_max_padded = x_max + padding_x
            y_lo -= padding_y
            y_hi += padding_y

            # Soft left-side padding
            if x_min_padded < 0:
                # If padding goes negative, we cap it at a small visible margin
                x_min_padded = -0.05 * x_max

            # Set the limits
            resid_ax.set_xlim(x_min_padded, x_max_padded)
            resid_ax.set_ylim(y_lo, y_hi)
            fit_ax.set_xlim(x_min_padded, x_max_padded)
            fit_ax.set_ylim(y_lo, y_hi)

        png_path = os.path.join(output_png_dir, f"{marker}_qc.png")
        fig.savefig(png_path, bbox_inches='tight')
        plt.close(fig)
        logger.info("🖼️ Saved %s", png_path)

    logger.info("✅ QC PNGs saved to %s", output_png_dir)





import logging



import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def run_denoising_pipeline_on_dataframe(
    df: pd.DataFrame,
    markers: list[str],
    morph_col: str = "Area",
    layer_suffix: str = "_ExclusionMembrane_Sum_Intensity",
    anchor_x: float = 15,
    anchor_y: float = 0,
    sd_multiplier: float = 3,
    signal_q_low: float = 95,
    signal_q_high: float = 99.99,
    min_cells_per_bin: int = 10,
    min_area: float = 15,
) -> dict:
    """
    Denoise intensity values per marker using a noise-based regression approach.

    Parameters:
    - df: DataFrame with morphological and intensity columns.
    - markers: List of marker prefixes to process.
    - morph_col: Column name for morphological metric (e.g., perimeter or area).
    - layer_suffix: Suffix for intensity columns in df.
    - anchor_x, anchor_y: Coordinates to anchor the noise regression line.
    - sd_multiplier: Multiplier for std dev to define high-morph cutoff.
    - signal_q_low, signal_q_high: Quantiles for selecting signal-range residuals.
    - min_cells_per_bin: Minimum cells required for regression.
    - min_area: Minimum morphological value to include.

    Returns:
    - dict with 'denoised_df' (DataFrame) and 'metadata' (per-marker info).
    """
    # Prepare morphological values (clipped by min_area)
    morph_vals = np.clip(df[morph_col].values, min_area, None)
    denoised_df = df.copy()
    metadata = {}

    for marker in markers:
        intensity_col = f"{marker}{layer_suffix}"
        if intensity_col not in df.columns:
            continue

        # 🚀 Step 1: Extract arrays directly from the DataFrame
        intensity = df[intensity_col].to_numpy(dtype=float)
        morph = morph_vals  # should already be a numpy array

        # 🚀 Step 2: Apply the valid mask (no DataFrame, just NumPy)
        valid_mask = (morph >= min_area) & (intensity > 0)
        area_filt = morph[valid_mask]
        inten_filt = intensity[valid_mask]

        if len(area_filt) < min_cells_per_bin:
            # Fill with zeros if not enough cells
            denoised_df[f"{marker}_ExclusionMembrane_FinalDenoised_Intensity"] = np.zeros_like(intensity)
            metadata[marker] = {
                "morph_col": morph_col,
                "morph_cutoff": np.nan,
                "noise_slope": np.nan,
                "anchor_x": anchor_x,
                "anchor_y": anchor_y,
                "area_reg_coef": np.nan,
                "area_reg_intercept": np.nan
            }
            continue

        # 🚀 Step 3: Compute the **intensity-based cutoff**
        # Sort by intensity, get corresponding morph values
        sort_idx = np.argsort(inten_filt)
        sorted_area = area_filt[sort_idx]

        # Compute the cutoff (mean + 3*SD) only on the sorted area values
        morph_mean = np.mean(sorted_area)
        morph_std = np.std(sorted_area)
        morph_cutoff = morph_mean + sd_multiplier * morph_std

        # 🚀 Step 4: Select the points above the cutoff
        high_mask = sorted_area > morph_cutoff

        if np.sum(high_mask) < min_cells_per_bin:
            noise_slope = 0.0
        else:
            # 🚀 Step 5: Fit the noise regression (clean and numpy-based)
            Xn = (sorted_area[high_mask] - anchor_x).reshape(-1, 1)
            yn = inten_filt[sort_idx][high_mask]
            noise_slope = float(LinearRegression(fit_intercept=False).fit(Xn, yn).coef_[0])

        # 🚀 Step 6: Compute noise fit and residuals
        noise_fit = np.where(
            morph > anchor_x,
            noise_slope * (morph - anchor_x) + anchor_y,
            0
        )

        residuals = intensity - noise_fit
        residuals_clipped = np.clip(residuals, 0, None)

        # 🚀 Step 7: Save to `denoised_df`
        denoised_df[f"{marker}_ExclusionMembrane_FinalDenoised_Intensity"] = residuals_clipped

        # 🚀 Step 8: Capture metadata
        metadata[marker] = {
            "morph_col": morph_col,
            "morph_cutoff": float(morph_cutoff),
            "noise_slope": noise_slope,
            "anchor_x": anchor_x,
            "anchor_y": anchor_y,
            "area_reg_coef": np.nan,
            "area_reg_intercept": np.nan
        }

    return {"denoised_df": denoised_df, "metadata": metadata}



def compute_denoised_reallocation_factors(protein_csv_paths, dirs, args):
    logger = logging.getLogger(__name__)
    morph_dir   = dirs["morph"]
    protein_dir = dirs["protein"]
    qc_out_dir  = dirs.get(
        "QC_metadata_denoised",
        os.path.join(dirs["QC"], "metadata_denoise")
    )
    os.makedirs(qc_out_dir, exist_ok=True)

    # Derive corresponding morphology paths
    morph_csv_paths = [
        os.path.join(morph_dir, os.path.basename(p))
        for p in protein_csv_paths
    ]

    all_fovs_data = []
    fov_ids       = []

    # Choose the morphological column
    if args.denoise_regress.lower() == "perimeter":
        morph_col = "Perimeter"
    elif args.denoise_regress.lower() == "area":
        morph_col = "Area"
    else:
        raise ValueError(
            f"Invalid args.denoise_regress: {args.denoise_regress}. "
            "Must be 'perimeter' or 'area'."
        )

    # Load and join each FOV
    for morph_path, protein_path in zip(morph_csv_paths, protein_csv_paths):
        fov_name = os.path.splitext(os.path.basename(protein_path))[0]
        try:
            morph_df   = pd.read_csv(morph_path)
            protein_df = pd.read_csv(protein_path)

            if morph_col not in morph_df.columns:
                raise KeyError(f"'{morph_col}' missing in {morph_path}")

            joint = morph_df[[morph_col]].join(protein_df, how="inner")
            joint["fov"] = fov_name

            all_fovs_data.append(joint)
            fov_ids.append(fov_name)

        except Exception as e:
            logger.warning(f"⚠️ Skipping {protein_path}: {e}")

    if not all_fovs_data:
        raise RuntimeError("❌ No valid FOVs for denoising!")

    full_df = pd.concat(all_fovs_data, ignore_index=True)
    # Identify markers by suffix
    layer_suffix = "_ExclusionMembrane_Sum_Intensity"
    intensity_cols = [
        c for c in full_df.columns
        if c.endswith(layer_suffix)
    ]
    markers = [
        c.replace(layer_suffix, "")
        for c in intensity_cols
    ]

    logger.info(
        "🚀 Running denoising on %d markers across %d FOVs",
        len(markers), len(fov_ids)
    )

    # 1) Run the core pipeline
    result       = run_denoising_pipeline_on_dataframe(
        df=full_df,
        markers=markers,
        morph_col=morph_col,
        layer_suffix=layer_suffix,
        anchor_x=args.signal_anchor_x if hasattr(args, "signal_anchor_x") else 15,
        anchor_y=args.signal_anchor_y if hasattr(args, "signal_anchor_y") else 0,
        # you can also pass other args here if desired
    )
    denoised_df  = result["denoised_df"]
    metadata     = result["metadata"]

    # 2) QC plotting (consumes metadata, does NOT re-run fits)
    qc_pdf = os.path.join(qc_out_dir, "denoiser_QC.pdf")
    qc_png = os.path.join(qc_out_dir, "denoiser_QC_pngs")
    save_signal_noise_qc_from_df(
        df=denoised_df,
        markers=markers,
        metadata=metadata,
        output_pdf=qc_pdf,
        output_png_dir=qc_png,
        morph_col=morph_col,
        layer_suffix=layer_suffix,
    )
    logger.info("✅ QC plots saved to %s/ (PDF + PNGs)", qc_out_dir)

    # 3) Write metadata table
    metadata_path = os.path.join(qc_out_dir, "denoiser_metadata.csv")
    try:
        metadata_df = pd.DataFrame.from_dict(metadata, orient="index")
        metadata_df.index.name = "marker"
        metadata_df.to_csv(metadata_path)
        logger.info("🧾 Saved denoiser metadata to %s", metadata_path)
    except Exception as e:
        logger.error("❌ Failed to save metadata: %s", e)

    # 🚀 Step 4) Persist denoised columns back to each FOV’s protein CSV
    for fov_name, group in denoised_df.groupby("fov"):
        out_path = os.path.join(protein_dir, f"{fov_name}.csv")
        if not os.path.exists(out_path):
            logger.warning("⚠️ Missing protein file for FOV %s, skipping", fov_name)
            continue

        try:
            prot_df = pd.read_csv(out_path)

            # 🚀 Drop any old denoised cols
            den_cols = [
                c for c in prot_df.columns
                if c.endswith("_ExclusionMembrane_FinalDenoised_Intensity")
            ]
            if den_cols:
                logger.info(f"🗑️ Dropping old denoised columns: {den_cols}")
                prot_df.drop(columns=den_cols, inplace=True)

            # 🚀 Append new denoised block
            new_block = group[
                [c for c in group.columns if c.endswith("_ExclusionMembrane_FinalDenoised_Intensity")]
            ].reset_index(drop=True)

            # 🚀 Alignment check
            if len(new_block) != len(prot_df):
                logger.warning(f"⚠️ Row count mismatch for {fov_name}: "
                               f"protein CSV ({len(prot_df)} rows) vs denoised block ({len(new_block)} rows).")
                continue

            # 🚀 Concatenate and save
            updated = pd.concat([prot_df.reset_index(drop=True), new_block], axis=1)
            updated.to_csv(out_path, index=False)
            logger.info(f"📝 Updated denoised values in {out_path}")

        except Exception as e:
            logger.error(f"❌ Could not update {out_path}: {e}")



