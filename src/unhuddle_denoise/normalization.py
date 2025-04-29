# src/unhuddle/normalization.py
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import logging
logger = logging.getLogger(__name__)


def adaptive_robust_scale_and_track(arr, initial_lower=1, initial_upper=99,
                                    min_range=1e-3, min_lower=0.01, max_upper=99.99,
                                    step=0.5):
    arr = arr.astype(float)
    arr_nonan = arr[np.isfinite(arr)]
    arr_nonzero = arr_nonan[arr_nonan != 0]

    lower = initial_lower
    upper = initial_upper

    while lower >= min_lower and upper <= max_upper:
        p1, p99 = np.percentile(arr_nonzero, [lower, upper])
        range_ = p99 - p1
        if range_ > min_range:
            scaled = (arr - p1) / range_
            return np.clip(scaled, 0, 1), (lower, upper, p1, p99)
        lower = max(min_lower, lower - step)
        upper = min(max_upper, upper + step)

    return np.zeros_like(arr), (None, None, None, None)


def total_intensity_normalize_and_scale(matrix, var_names,
                                        lower=1, upper=99, min_range=1e-3):
    """
    Performs per-cell normalization using total summed intensity, followed by adaptive robust per-marker scaling.

    Parameters:
        matrix (np.ndarray): Raw intensity matrix (cells x markers)
        var_names (list): Marker names (len = matrix.shape[1])
        lower, upper: Initial percentiles for robust scaling
        min_range: Min dynamic range before loosening percentiles

    Returns:
        scaled_matrix (np.ndarray): Matrix after per-cell and per-marker normalization
        cell_total (np.ndarray): Per-cell total intensity (scale used)
    """
    cell_total = matrix.sum(axis=1)
    norm_matrix = matrix / np.where(cell_total[:, None] == 0, np.nan, cell_total[:, None])

    scaled_matrix = np.zeros_like(norm_matrix)

    for j in range(norm_matrix.shape[1]):
        col = norm_matrix[:, j]
        nonzero_mask = col != 0

        if np.any(nonzero_mask):
            scaled_col, (lower_p, upper_p, p1, p99) = adaptive_robust_scale_and_track(
                col, initial_lower=lower, initial_upper=upper, min_range=min_range
            )
            scaled_col[~nonzero_mask] = 0
            scaled_matrix[:, j] = scaled_col
            logger.debug(f"[{var_names[j]}] adaptive scaling {lower_p}-{upper_p}% → [{p1:.3g}, {p99:.3g}]")
        else:
            logger.warning(f"[{var_names[j]}] No nonzero values; filling 0")
            scaled_matrix[:, j] = 0

    return scaled_matrix, cell_total



def compute_normalized_intensities_for_fov(fov_folder, corrected_sum_df, sensor_markers):
    """
    Computes per-cell and per-marker normalized intensities using sensor markers.

    Args:
        fov_folder (str): Path to the FOV folder (used for naming only).
        corrected_sum_df (pd.DataFrame): DataFrame with 'Label' and marker columns.
        sensor_markers (list): List of markers used for per-cell normalization.

    Returns:
        fov_name (str): Extracted FOV name
        normalized_df (pd.DataFrame): DataFrame with 'Label', scaled marker values
    """
    import numpy as np
    import pandas as pd
    import os

    fov_name = os.path.basename(fov_folder)
    logger.info(f"Normalizing intensities for {fov_name}")

    if "Label" not in corrected_sum_df.columns:
        logger.error(f"{fov_name} is missing required 'Label' column.")
        return fov_name, None

    # Extract marker columns
    marker_columns = sorted([col for col in corrected_sum_df.columns if col != 'Label'])
    raw_matrix = corrected_sum_df[marker_columns].values
    logger.debug(f"Using marker columns: {marker_columns}")

    scaled_matrix, total_intensity = total_intensity_normalize_and_scale(
        matrix=raw_matrix,
        var_names=marker_columns
    )

    # Construct output DataFrame
    normalized_df = pd.DataFrame(scaled_matrix, columns=marker_columns)
    normalized_df.insert(0, 'Label', corrected_sum_df["Label"].values)

    return fov_name, normalized_df
