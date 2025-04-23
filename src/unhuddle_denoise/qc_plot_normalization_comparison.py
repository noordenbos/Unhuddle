import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
logger = logging.getLogger(__name__)

def plot_sensor_marker_counts(marker_counts, output_path):
    """
    Histogram of number of sensor markers used per cell.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(6, 4))
    plt.hist(marker_counts, bins=np.arange(0, 7)-0.5, color='skyblue', edgecolor='black')
    plt.xlabel("Sensor Markers Used per Cell")
    plt.ylabel("Cell Count")
    plt.title("Sensor Marker Count Distribution")
    plt.xticks(range(0, 6))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_intensity_compression(raw_df, norm_df_weighted, norm_df_area, marker_subset, output_path):
    """
    Scatter plot comparing raw summed intensity vs normalized (weighted and area-based).
    """
    import matplotlib.pyplot as plt

    sum_raw = raw_df[marker_subset].sum(axis=1)
    sum_weighted = norm_df_weighted[marker_subset].sum(axis=1)
    sum_area = norm_df_area[marker_subset].sum(axis=1)

    plt.figure(figsize=(6, 6))
    plt.scatter(sum_raw, sum_weighted, s=1, alpha=0.3, label="Weighted Norm")
    plt.scatter(sum_raw, sum_area, s=1, alpha=0.3, label="Raw / Area")
    plt.plot([sum_raw.min(), sum_raw.max()], [sum_raw.min(), sum_raw.max()], 'k--')
    plt.xlabel("Raw Total Intensity")
    plt.ylabel("Normalized Total Intensity")
    plt.title("Total Intensity Compression")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_intensity_compression(raw_df, norm_df_weighted, norm_df_area, marker_subset, output_path):
    """
    Scatter plot comparing raw summed intensity vs normalized (weighted and area-based).
    """
    import matplotlib.pyplot as plt

    sum_raw = raw_df[marker_subset].sum(axis=1)
    sum_weighted = norm_df_weighted[marker_subset].sum(axis=1)
    sum_area = norm_df_area[marker_subset].sum(axis=1)

    plt.figure(figsize=(6, 6))
    plt.scatter(sum_raw, sum_weighted, s=1, alpha=0.3, label="Weighted Norm")
    plt.scatter(sum_raw, sum_area, s=1, alpha=0.3, label="Raw / Area")
    plt.plot([sum_raw.min(), sum_raw.max()], [sum_raw.min(), sum_raw.max()], 'k--')
    plt.xlabel("Raw Total Intensity")
    plt.ylabel("Normalized Total Intensity")
    plt.title("Total Intensity Compression")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_marker_correlation_heatmaps(raw_df, norm_df_weighted, norm_df_area, marker_subset, output_path):
    """
    Three heatmaps side-by-side showing marker-to-marker correlations.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sns.heatmap(raw_df[marker_subset].corr(), ax=axes[0], cmap="vlag", center=0)
    axes[0].set_title("Correlation – Raw")

    sns.heatmap(norm_df_weighted[marker_subset].corr(), ax=axes[1], cmap="vlag", center=0)
    axes[1].set_title("Correlation – Weighted")

    sns.heatmap(norm_df_area[marker_subset].corr(), ax=axes[2], cmap="vlag", center=0)
    axes[2].set_title("Correlation – Raw / Area")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_marker_distribution_storyboard(raw_df, norm_df_weighted, norm_df_area, marker_subset, output_path):
    """
    Storyboard of overlaid KDE plots per marker: raw, weighted norm, and area norm.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math

    n = len(marker_subset)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    for i, marker in enumerate(marker_subset):
        ax = axes[i]
        sns.kdeplot(raw_df[marker], label="Raw", ax=ax, linestyle='--', warn_singular=False)
        sns.kdeplot(norm_df_weighted[marker], label="Weighted", ax=ax, warn_singular=False)
        sns.kdeplot(norm_df_area[marker], label="Raw / Area", ax=ax, warn_singular=False)

        ax.set_title(marker)
        ax.set_xlabel("Intensity")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Per-Marker Intensity Distribution", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path)
    plt.close()

