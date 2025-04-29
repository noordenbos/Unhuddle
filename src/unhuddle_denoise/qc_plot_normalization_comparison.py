import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math

def qc_plot_normalization_comparison(
    adata,
    raw_layer="sum_unhuddle_denoised",
    scaled_layer="sum_unhuddle_denoised_scaled",
    area_key="Area",
    marker_subset=None,
    pdf_all_rows_path="qc_all_cells_vs_area.pdf",
    pdf_storyboard_path="qc_storyboard_per_marker.pdf"
):
    raw_mat = adata.layers[raw_layer]
    scaled_mat = adata.layers[scaled_layer]
    area = adata.obs[area_key].values

    if marker_subset is None:
        marker_subset = adata.var_names.tolist()

    marker_idx = [i for i, name in enumerate(adata.var_names) if name in marker_subset]
    area_safe = np.where(area == 0, np.nan, area)

    with PdfPages(pdf_all_rows_path) as pdf:
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
            pdf.savefig(fig)
            plt.close()

    with PdfPages(pdf_storyboard_path) as pdf:
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
        pdf.savefig(fig)
        plt.close()

    return "✅ PDF outputs created: 3-panel per cell + marker storyboard."

