import os
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def _per_cell_normalize(matrix: np.ndarray,
                        var_names: list,
                        sensor_markers: list | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize each cell either by total intensity or by sum over sensor_markers.
    Returns (norm_matrix, cell_total).
    """
    if sensor_markers:
        idx = [var_names.index(m) for m in sensor_markers if m in var_names]
        if not idx:
            logger.warning("No sensor markers found; falling back to total intensity")
            cell_total = matrix.sum(axis=1)
        else:
            cell_total = matrix[:, idx].sum(axis=1)
    else:
        cell_total = matrix.sum(axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        norm = matrix / cell_total[:, None]
    return norm, cell_total

def compute_cohort_marker_stats(all_matrices: list[np.ndarray],
                                var_names: list[str],
                                *,
                                lower_pct: float = 1.0,
                                upper_pct: float = 99.0,
                                min_range: float = 1e-3
                               ) -> pd.DataFrame:
    """
    all_matrices : list of (cells × markers) arrays from each FOV after per-cell norm
    var_names    : list of marker names
    Returns a DataFrame with columns [marker, p1, p99] computed cohort-wide.
    """
    # Stack all FOVs together (might be big – you can also compute per-chunk)
    big = np.vstack(all_matrices)  # shape = (total_cells, n_markers)

    records = []
    for j, m in enumerate(var_names):
        col = big[:, j]
        nonzero = col[col > 0]
        if nonzero.size == 0:
            p1 = p99 = 0.0
        else:
            p1, p99 = np.percentile(nonzero, [lower_pct, upper_pct])
            if (p99 - p1) < min_range:
                p1, p99 = nonzero.min(), nonzero.max()
        records.append({"marker": m, "p1": float(p1), "p99": float(p99)})
    return pd.DataFrame(records)

def apply_cohort_scaling(norm_matrix: np.ndarray,
                         var_names: list[str],
                         stats_df: pd.DataFrame
                        ) -> np.ndarray:
    """
    norm_matrix : (cells × markers) per-cell-normalized array
    stats_df    : output of compute_cohort_marker_stats
    Returns scaled_matrix in [0,1] using the fixed p1/p99 for each marker.
    """
    scaled = np.zeros_like(norm_matrix)
    for j, m in enumerate(var_names):
        p1 = stats_df.loc[stats_df["marker"] == m, "p1"].iloc[0]
        p99 = stats_df.loc[stats_df["marker"] == m, "p99"].iloc[0]
        col = norm_matrix[:, j]
        with np.errstate(divide='ignore', invalid='ignore'):
            if (p99 - p1) >= 1e-3:
                s = (col - p1) / (p99 - p1)
            else:
                s = np.zeros_like(col)
        s = np.clip(s, 0.0, 1.0)
        s[col == 0] = 0.0
        scaled[:, j] = s
    return scaled


def _adaptive_marker_scaling(col: np.ndarray,
                             min_range: float = 1e-3,
                             cv_thresh: float = 0.1,
                             cv_frac_thresh: float = 0.10
                            ) -> tuple[np.ndarray, dict]:
    """
    Scale a single marker column using CV gating + sparsity-aware robust scaling:
      1. If no non-zero values -> all zeros.
      2. Compute fraction_nonzero and CV on nonzeros.
      3. If (fraction_nonzero < cv_frac_thresh AND CV < cv_thresh) -> binary presence/absence.
      4. Else pick percentiles:
           >30% nonzero -> 1–99%
           5–30%  -> 5–95%
           <5%    -> 10–90%
      5. If (p99 - p1) < min_range -> fallback to true min/max.
      6. Scale and clip to [0,1], preserving original zeros.

    Returns (scaled_col, stats_dict).
    """
    # 1) collect nonzeros
    nonzero = col[col > 0]
    n = len(col)
    if nonzero.size == 0:
        return np.zeros_like(col), {
            "marker_method": "all_zero",
            "fraction_nonzero": 0.0
        }

    # 2) compute fraction and CV
    frac = nonzero.size / n
    mean = nonzero.mean()
    std  = nonzero.std()
    cv   = std / mean if mean > 0 else np.inf

    # binary gate if sparse OR low CV
    if frac < cv_frac_thresh or cv < cv_thresh:
        scaled = (col > 0).astype(float)
        return scaled, {
            "marker_method": "binary_by_cv_or_frac",
            "fraction_nonzero": frac,
            "cv": float(cv)
        }

    # 4) percentile-based robust scaling
    if frac > 0.30:
        lower_pct, upper_pct = 0.1, 99.9
    elif frac > 0.05:
        lower_pct, upper_pct = 1.0, 99.0
    else:
        lower_pct, upper_pct = 10.0, 90.0

    p1, p99 = np.percentile(nonzero, [lower_pct, upper_pct])
    if (p99 - p1) < min_range:
        p1, p99 = nonzero.min(), nonzero.max()

    # 5) scale
    with np.errstate(divide='ignore', invalid='ignore'):
        if (p99 - p1) >= min_range:
            scaled = (col - p1) / (p99 - p1)
        else:
            scaled = np.zeros_like(col)

    # 6) clip and maintain zeros
    scaled = np.clip(scaled, 0.0, 1.0)
    scaled[col == 0] = 0.0

    stats = {
        "marker_method": "robust",
        "fraction_nonzero": frac,
        "p_lower_pct": lower_pct,
        "p_upper_pct": upper_pct,
        "p1": float(p1),
        "p99": float(p99)
    }
    return scaled, stats


def total_intensity_normalize_and_scale(matrix: np.ndarray,
                                        var_names: list[str],
                                        sensor_markers: list[str] | None = None,
                                        min_range: float = 1e-3,
                                        cv_thresh: float = 0.1,
                                        cv_frac_thresh: float = 0.10
                                       ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Full per-cell then per-marker normalization with adaptive CV-gated scaling.

    Returns:
      scaled_matrix : (cells × markers) numpy array in [0,1]
      cell_total    : the per-cell normalizer used
      stats_df      : diagnostics per marker
    """
    norm_matrix, cell_total = _per_cell_normalize(matrix, var_names, sensor_markers)
    scaled = np.zeros_like(norm_matrix)
    records = []
    for j, var in enumerate(var_names):
        col = norm_matrix[:, j]
        sc, stats = _adaptive_marker_scaling(col,
                                             min_range=min_range,
                                             cv_thresh=cv_thresh,
                                             cv_frac_thresh=cv_frac_thresh)
        scaled[:, j] = sc
        stats["marker"] = var
        records.append(stats)
        logger.debug(f"{var}: {stats}")

    stats_df = pd.DataFrame(records)
    stats_df = stats_df[["marker", *[c for c in stats_df.columns if c != "marker"]]]
    return scaled, cell_total, stats_df


def compute_normalized_intensities_for_fov(fov_folder: str,
                                           corrected_sum_df: pd.DataFrame,
                                           sensor_markers: list[str] | None = None,
                                           min_range: float = 1e-3,
                                           cv_thresh: float = 1.0,
                                           cv_frac_thresh: float = 0.10
                                          ) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    Args:
      fov_folder       Path (basename used)
      corrected_sum_df DataFrame with 'Label' + marker columns
      sensor_markers   List of markers for per-cell norm

    Returns:
      fov_name      Basename
      normalized_df DataFrame with 'Label' + scaled
      stats_df      Diagnostics
    """
    fov_name = os.path.basename(fov_folder)
    logger.info(f"Normalizing intensities for {fov_name}")

    if "Label" not in corrected_sum_df.columns:
        logger.error(f"{fov_name} missing 'Label'.")
        return fov_name, None, None

    marker_cols = sorted(c for c in corrected_sum_df.columns if c != "Label")
    raw = corrected_sum_df[marker_cols].values

    scaled_matrix, cell_total, stats_df = total_intensity_normalize_and_scale(
        raw, marker_cols,
        sensor_markers=sensor_markers,
        min_range=min_range,
        cv_thresh=cv_thresh,
        cv_frac_thresh=cv_frac_thresh
    )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"\n[Normalization stats for {fov_name}]\n{stats_df.to_string(index=False)}")

    normalized_df = pd.DataFrame(scaled_matrix, columns=marker_cols)
    normalized_df.insert(0, "Label", corrected_sum_df["Label"].values)

    return fov_name, normalized_df, stats_df



