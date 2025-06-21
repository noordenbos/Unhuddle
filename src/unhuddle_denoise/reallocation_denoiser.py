#script currently deprecated, has been replaced by percentile_denoise.py

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
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression

def save_signal_noise_qc_from_df(
    df: pd.DataFrame,
    markers: list[str],
    metadata: dict[str, dict],
    output_pdf: str = "signal_noise_qc.pdf",
    output_png: str = "signal_noise_qc.png",
    morph_col: str = "Area",
    layer_suffix: str = "_ExclusionMembrane_Sum_Intensity",
    cols: int = 2,
    signal_q_low: float = 99,
    signal_q_high: float = 99.99,
) -> None:
    logger = logging.getLogger("unhuddle")
    logger.info("📊 Starting noise QC plotting for %d markers...", len(markers))

    n_rows = int(np.ceil(len(markers) / cols))
    fig, axs = plt.subplots(n_rows, cols * 2, figsize=(cols * 10, n_rows * 5), dpi=150)
    axs = axs.flatten()

    for idx, marker in enumerate(markers):
        fit_ax = axs[2 * idx]
        resid_ax = axs[2 * idx + 1]
        colname = f"{marker}{layer_suffix}"
        meta = metadata.get(marker, {})

        if colname not in df.columns or morph_col not in df.columns or not meta:
            fit_ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=fit_ax.transAxes)
            resid_ax.axis('off')
            continue

        # Extract metadata explicitly
        anchor_x = meta.get("anchor_x", 0)
        anchor_y = meta.get("anchor_y", 0)
        noise_slope = meta["noise_slope"]
        morph_cutoff = meta["morph_cutoff"]
        noise_q = meta["noise_q"]

        area = df[morph_col].values
        inten = df[colname].values

        # Calculate noise fit and residuals explicitly
        noise_fit = np.where(area > anchor_x, noise_slope * (area - anchor_x) + anchor_y, 0)
        residuals = inten - noise_fit

        # Define masks
        area_above_cutoff = area > morph_cutoff
        if np.sum(area_above_cutoff) == 0:
            noise_mask = np.zeros_like(area, dtype=bool)
        else:
            intensity_threshold = np.percentile(inten[area_above_cutoff], noise_q)
            noise_mask = area_above_cutoff & (inten <= intensity_threshold)

        signal_mask = (inten > np.percentile(inten, signal_q_low)) & (inten < np.percentile(inten, signal_q_high))

        # Plot intensity vs area
        fit_ax.scatter(area, inten, color='lightgrey', alpha=0.3, s=5, label='All points')
        fit_ax.scatter(area[noise_mask], inten[noise_mask], color='blue', alpha=0.5, s=5, label='Noise points')
        fit_ax.scatter(area[signal_mask], inten[signal_mask], color='orange', alpha=0.7, s=5, label='Signal points')

        # Plot noise line
        xs = np.linspace(area.min(), area.max(), 200)
        ys_noise = np.where(xs > anchor_x, noise_slope * (xs - anchor_x) + anchor_y, 0)
        fit_ax.plot(xs, ys_noise, color='green', lw=2, label="Noise fit")

        # Fit and plot signal line
        x_signal = area[signal_mask]
        y_signal = inten[signal_mask]
        if len(x_signal) < 10:
            logger.warning(f"⚠️ Skipping {marker}: too few signal points")
            continue

        signal_fit = LinearRegression(fit_intercept=False).fit((x_signal - 10).reshape(-1, 1), y_signal)
        signal_slope = float(signal_fit.coef_[0])
        ys_signal = signal_slope * (xs - 10)
        fit_ax.plot(xs, ys_signal, color='red', lw=2, label=f"Signal fit (indicative)")

        # Plot cutoff line
        fit_ax.axvline(morph_cutoff, linestyle='--', color='purple', label="Noise cutoff")

        # Compute shared Y-axis limits (combined intensity and residuals)
        combined_values = np.concatenate([inten, residuals])
        y_lo, y_hi = np.percentile(combined_values, [1, 99.99])
        if np.isclose(y_lo, y_hi):
            y_lo -= 0.5
            y_hi += 0.5
        y_margin = (y_hi - y_lo) * 0.1
        y_lo -= y_margin
        y_hi += y_margin

        # Set shared Y-axis limits
        fit_ax.set_ylim(y_lo, y_hi)
        resid_ax.set_ylim(y_lo, y_hi)

        # Add grey rectangle for noise estimation area (after setting ylim)
        fit_ax.add_patch(patches.Rectangle((morph_cutoff, y_lo),
                                           fit_ax.get_xlim()[1] - morph_cutoff,
                                           y_hi - y_lo,
                                           color='lightgrey', alpha=0.2))

        fit_ax.set_title(f"{marker} Intensity vs {morph_col}")
        fit_ax.set_xlabel(morph_col)
        fit_ax.set_ylabel("Intensity")
        fit_ax.legend(fontsize=6)

        # Plot residuals (all points blue)
        resid_ax.scatter(area, residuals, color='blue', alpha=0.5, s=5, label='Residuals')
        resid_ax.axhline(0, color='red', linestyle='--')

        resid_ax.set_xlim(area.min(), area.max())
        resid_ax.set_title(f"{marker} Residuals")
        resid_ax.set_xlabel(morph_col)
        resid_ax.set_ylabel("Residual Intensity")
        resid_ax.legend(fontsize=6)

    # Hide unused axes
    for j in range(2 * len(markers), len(axs)):
        axs[j].axis('off')

    plt.tight_layout()

    # Save plots
    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    logger.info("✅ QC PDF saved to %s", output_pdf)

    fig.savefig(output_png, bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.info("✅ QC PNG saved to %s", output_png)





import logging



import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def run_denoising_pipeline_on_dataframe(
    df: pd.DataFrame,
    markers: list[str],
    morph_col: str = "Area",
    layer_suffix: str = "_ExclusionMembrane_Sum_Intensity",
    anchor_x: float = 0,
    anchor_y: float = 0,
    sd_multiplier: float = 3,
    x_anchor_multiplier: float = -1.0,
    signal_q_low: float = 95,
    signal_q_high: float = 99.99,
    min_cells_per_bin: int = 10,
    min_area: float = 15,
    noise_q: float = 75,  # percentile noise in large cells
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
    - x_anchor_multiplier: Multiplier for std dev to calculate x-anchor. anchor_x = mean + multiplier * std.
    - signal_q_low, signal_q_high: Quantiles for selecting signal-range residuals.
    - min_cells_per_bin: Minimum cells required for regression.
    - min_area: Minimum morphological value to include.
    - noise_q: Percentile cutoff for selecting noise points (default 95).

    Returns:
    - dict with 'denoised_df' (DataFrame) and 'metadata' (per-marker info).
    """
    morph_vals = np.clip(df[morph_col].values, min_area, None)
    denoised_df = df.copy()
    metadata = {}

    for marker in markers:
        intensity_col = f"{marker}{layer_suffix}"
        if intensity_col not in df.columns:
            continue

        intensity = df[intensity_col].to_numpy(dtype=float)
        morph = morph_vals

        valid_mask = (morph >= min_area) & (intensity > 0)
        area_filt = morph[valid_mask]
        inten_filt = intensity[valid_mask]

        if len(area_filt) < min_cells_per_bin:
            denoised_df[f"{marker}{layer_suffix}_denoised"] = np.zeros_like(intensity)
            metadata[marker] = {
                "morph_col": morph_col,
                "morph_cutoff": np.nan,
                "noise_slope": np.nan,
                "anchor_x": anchor_x,
                "anchor_y": anchor_y,
                "noise_q": noise_q,
                "x_anchor_multiplier": x_anchor_multiplier,
            }
            continue

        sort_idx = np.argsort(inten_filt)
        sorted_area = area_filt[sort_idx]

        morph_mean = np.mean(sorted_area)
        morph_std = np.std(sorted_area)
        # Calculate anchor_x using the configurable multiplier
        anchor_x = morph_mean + x_anchor_multiplier * morph_std
        anchor_y = 0
        morph_cutoff = morph_mean + sd_multiplier * morph_std

        # 🚀 Updated Step: Select points above morph_cutoff first
        high_morph_mask = area_filt > morph_cutoff
        if np.sum(high_morph_mask) < min_cells_per_bin:
            noise_slope = 0.0
        else:
            # 🚀 Updated Step: Within high morph points, select bottom noise_q percentile by intensity
            high_morph_intensities = inten_filt[high_morph_mask]
            intensity_threshold = np.percentile(high_morph_intensities, noise_q)
            noise_mask = high_morph_mask & (inten_filt <= intensity_threshold)

            if np.sum(noise_mask) < min_cells_per_bin:
                noise_slope = 0.0
            else:
                Xn = (area_filt[noise_mask] - anchor_x).reshape(-1, 1)
                yn = inten_filt[noise_mask]
                noise_slope = float(LinearRegression(fit_intercept=False).fit(Xn, yn).coef_[0])

        noise_fit = np.where(
            morph > anchor_x,
            noise_slope * (morph - anchor_x) + anchor_y,
            0
        )

        residuals = intensity - noise_fit
        residuals_clipped = np.clip(residuals, 0, None)

        denoised_df[f"{marker}{layer_suffix}_denoised"] = residuals_clipped

        metadata[marker] = {
            "morph_col": morph_col,
            "morph_cutoff": float(morph_cutoff),
            "noise_slope": noise_slope,
            "anchor_x": anchor_x,
            "anchor_y": anchor_y,
            "noise_q": noise_q,
            "x_anchor_multiplier": x_anchor_multiplier,
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
        anchor_x=args.signal_anchor_x if hasattr(args, "signal_anchor_x") else 0,
        anchor_y=args.signal_anchor_y if hasattr(args, "signal_anchor_y") else 0,
        sd_multiplier=getattr(args, "denoise_sd_multiplier", 3.0),
        x_anchor_multiplier=getattr(args, "denoise_x_anchor_multiplier", -1.0),
        signal_q_low=getattr(args, "denoise_signal_q_low", 95.0),
        signal_q_high=getattr(args, "denoise_signal_q_high", 99.99),
        min_cells_per_bin=getattr(args, "denoise_min_cells_per_bin", 10),
        min_area=getattr(args, "denoise_min_area", 15.0),
        noise_q=getattr(args, "denoise_noise_q", 75.0),
    )
    denoised_df  = result["denoised_df"]
    metadata     = result["metadata"]

    # 2) QC plotting (consumes metadata, does NOT re-run fits)
    qc_pdf = os.path.join(qc_out_dir, "noise_cone_denoiser_QC.pdf")
    qc_png = os.path.join(qc_out_dir, "noise_cone_denoiser_QC.png")
    save_signal_noise_qc_from_df(
        df=denoised_df,
        markers=markers,
        metadata=metadata,
        output_pdf=qc_pdf,
        output_png=qc_png,
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

    # 🚀 Step 4) Persist denoised columns back to each FOV's protein CSV
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
                if c.endswith("_denoised")
            ]
            if den_cols:
                logger.info(f"🗑️ Dropping old denoised columns: {den_cols}")
                prot_df.drop(columns=den_cols, inplace=True)

            # 🚀 Append new denoised block
            new_block = group[
                [c for c in group.columns if c.endswith("_denoised")]
            ].reset_index(drop=True)

            logger.debug(f"🔍 Denoised columns for {fov_name}: {list(new_block.columns)}")
            logger.debug(f"🔍 New block shape for {fov_name}: {new_block.shape}")
            logger.debug(f"🔍 Group shape for {fov_name}: {group.shape}")
            logger.debug(f"🔍 Prot_df shape for {fov_name}: {prot_df.shape}")
            
            logger.info(f"🔍 Processing {fov_name}: {len(new_block.columns)} denoised columns, {new_block.shape[0]} rows")

            # 🚀 Compute Mean intensity columns from Sum intensity
            area_col = "Area"
            memexcl_area_col = "ExclusionMembrane_Area"
            
            if memexcl_area_col in prot_df.columns:
                # Use exclusion membrane areas for mean intensity calculation
                memexcl_area_values = prot_df[memexcl_area_col].values
                valid_memexcl_areas = np.where(memexcl_area_values > 0, memexcl_area_values, 1)
                
                logger.debug(f"🔍 Exclusion membrane area values for {fov_name}: min={memexcl_area_values.min()}, max={memexcl_area_values.max()}, mean={memexcl_area_values.mean()}")
                logger.debug(f"🔍 Valid exclusion membrane areas for {fov_name}: min={valid_memexcl_areas.min()}, max={valid_memexcl_areas.max()}, mean={valid_memexcl_areas.mean()}")
                
                sum_intensity_cols = [c for c in new_block.columns if c.endswith("_ExclusionMembrane_Sum_Intensity_denoised")]
                logger.debug(f"🔍 Found {len(sum_intensity_cols)} sum intensity columns: {sum_intensity_cols}")
                logger.info(f"🔍 Found {len(sum_intensity_cols)} sum intensity columns for mean computation in {fov_name}")
                
                for col in new_block.columns:
                    if col.endswith("_ExclusionMembrane_Sum_Intensity_denoised"):
                        marker = col.replace("_ExclusionMembrane_Sum_Intensity_denoised", "")
                        sum_values = new_block[col].values
                        mean_col = f"{marker}_ExclusionMembrane_Mean_Intensity_denoised"
                        mean_values = sum_values / valid_memexcl_areas
                        new_block[mean_col] = mean_values
                        logger.debug(f"✅ Computed {mean_col} from {col}: sum_min={sum_values.min()}, sum_max={sum_values.max()}, mean_min={mean_values.min()}, mean_max={mean_values.max()}")
                        logger.info(f"✅ Computed {mean_col} from {col} for {fov_name}")
                    else:
                        logger.debug(f"⏭️ Skipping {col} (not a sum intensity column)")
            elif area_col in prot_df.columns:
                # Fallback to cell areas if exclusion membrane areas not available
                area_values = prot_df[area_col].values
                valid_areas = np.where(area_values > 0, area_values, 1)
                
                logger.warning(f"⚠️ ExclusionMembrane_Area not found in {out_path}, using cell areas as fallback")
                logger.debug(f"🔍 Area values for {fov_name}: min={area_values.min()}, max={area_values.max()}, mean={area_values.mean()}")
                
                sum_intensity_cols = [c for c in new_block.columns if c.endswith("_ExclusionMembrane_Sum_Intensity_denoised")]
                logger.debug(f"🔍 Found {len(sum_intensity_cols)} sum intensity columns: {sum_intensity_cols}")
                logger.info(f"🔍 Found {len(sum_intensity_cols)} sum intensity columns for mean computation in {fov_name}")
                
                for col in new_block.columns:
                    if col.endswith("_ExclusionMembrane_Sum_Intensity_denoised"):
                        marker = col.replace("_ExclusionMembrane_Sum_Intensity_denoised", "")
                        sum_values = new_block[col].values
                        mean_col = f"{marker}_ExclusionMembrane_Mean_Intensity_denoised"
                        mean_values = sum_values / valid_areas
                        new_block[mean_col] = mean_values
                        logger.debug(f"✅ Computed {mean_col} from {col}: sum_min={sum_values.min()}, sum_max={sum_values.max()}, mean_min={mean_values.min()}, mean_max={mean_values.max()}")
                        logger.info(f"✅ Computed {mean_col} from {col} for {fov_name}")
                    else:
                        logger.debug(f"⏭️ Skipping {col} (not a sum intensity column)")
            else:
                logger.warning(f"⚠️ Neither Area nor ExclusionMembrane_Area column found in {out_path}, skipping mean intensity computation")
                logger.debug(f"🔍 Available columns in prot_df: {list(prot_df.columns)}")

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



