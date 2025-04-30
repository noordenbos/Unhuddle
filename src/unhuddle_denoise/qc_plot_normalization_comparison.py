import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def qc_plot_normalization_comparison(
    adata,
    raw_layer="sum_unhuddle_denoised",
    scaled_layer="sum_unhuddle_denoised_scaled",
    area_key="Area",
    marker_subset=None,
    pdf_all_rows_path="qc_all_cells_vs_area.pdf",     # only used to name the png folder
    pdf_storyboard_path="qc_storyboard_per_marker.pdf"  # only used to name the png file
):
    raw_mat = adata.layers[raw_layer]
    scaled_mat = adata.layers[scaled_layer]
    area = adata.obs[area_key].values
    area_safe = np.where(area == 0, np.nan, area)

    if marker_subset is None:
        marker_subset = adata.var_names.tolist()

    marker_idx = [i for i, name in enumerate(adata.var_names) if name in marker_subset]

    # === Individual PNGs per marker ===
    png_dir = os.path.splitext(pdf_all_rows_path)[0] + "_png"
    os.makedirs(png_dir, exist_ok=True)

    for j in marker_idx:
        name = adata.var_names[j]
        raw = raw_mat[:, j]
        raw_area_norm = raw / area_safe
        scaled = scaled_mat[:, j]

        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=False)

        axs[0].scatter(area, raw, s=2, alpha=0.3)
        axs[0].set_title(f"{name} - Raw")
        axs[0].set_xlabel("Area")
        axs[0].set_ylabel("Intensity")

        axs[1].scatter(area, raw_area_norm, s=2, alpha=0.3)
        axs[1].set_title(f"{name} - Raw / Area")
        axs[1].set_xlabel("Area")

        axs[2].scatter(area, scaled, s=2, alpha=0.3)
        axs[2].set_title(f"{name} - Robust Scaled")
        axs[2].set_xlabel("Area")

        fig.suptitle(f"{name} normalization comparison")
        plt.tight_layout()
        fig.savefig(os.path.join(png_dir, f"{name}_normalization.png"), dpi=200)
        plt.close()

    logger.info(f"✅ Saved per-marker normalization PNGs to: {png_dir}")

    # === Storyboard PNG ===
    n = len(marker_idx)
    fig, axs = plt.subplots(n, 3, figsize=(10, 3 * n), sharex='col')

    for i, j in enumerate(marker_idx):
        name = adata.var_names[j]
        raw = raw_mat[:, j]
        raw_area_norm = raw / area_safe
        scaled = scaled_mat[:, j]

        axs[i, 0].scatter(area, raw, s=2, alpha=0.3)
        axs[i, 0].set_ylabel(name)

        axs[i, 1].scatter(area, raw_area_norm, s=2, alpha=0.3)
        axs[i, 2].scatter(area, scaled, s=2, alpha=0.3)

    axs[0, 0].set_title("Raw")
    axs[0, 1].set_title("Raw / Area")
    axs[0, 2].set_title("Robust Scaled")

    for ax in axs[-1, :]:
        ax.set_xlabel("Area")

    plt.tight_layout()
    storyboard_path = os.path.splitext(pdf_storyboard_path)[0] + ".png"
    fig.savefig(storyboard_path, dpi=200)
    plt.close()

    logger.info(f"✅ Saved storyboard PNG: {storyboard_path}")
    return f"✅ PNGs created: per-marker in {png_dir}, storyboard at {storyboard_path}"
