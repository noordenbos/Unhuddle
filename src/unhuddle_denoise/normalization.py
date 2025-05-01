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




def compute_adaptive_marker_stats_from_cohort(
    all_matrices: list[np.ndarray],
    var_names: list[str],
    sample_max_cells: int = 100_000,
    min_range: float = 1e-3,
    cv_thresh: float = 0.1,
    cv_frac_thresh: float = 0.10
) -> pd.DataFrame:
    """
    Cohort-wide marker-wise scaling strategy with adaptive fallback based on sparsity and CV.

    Parameters
    ----------
    all_matrices : list of np.ndarray
        List of (cells x markers) matrices (already per-cell normalized).
    var_names : list of str
        Marker names.
    sample_max_cells : int
        Maximum number of cells to use for cohort-wide statistics.
    min_range : float
        Minimum dynamic range for robust scaling.
    cv_thresh : float
        CV threshold for triggering binary fallback.
    cv_frac_thresh : float
        Sparsity threshold for triggering binary fallback.

    Returns
    -------
    pd.DataFrame with columns:
      marker, method, fraction_nonzero, CV, p1, p99, p_lower_pct, p_upper_pct
    """

    # Stack all matrices, with optional downsampling
    stacked = np.vstack(all_matrices)
    if stacked.shape[0] > sample_max_cells:
        idx = np.random.choice(stacked.shape[0], size=sample_max_cells, replace=False)
        stacked = stacked[idx]

    records = []
    for j, marker in enumerate(var_names):
        col = stacked[:, j]
        nonzero = col[col > 0]
        n = len(col)

        if nonzero.size == 0:
            records.append({
                "marker": marker,
                "marker_method": "all_zero",
                "fraction_nonzero": 0.0,
                "cv": None,
                "p_lower_pct": None,
                "p_upper_pct": None,
                "p1": 0.0,
                "p99": 0.0
            })
            continue

        frac = nonzero.size / n
        mean = nonzero.mean()
        std = nonzero.std()
        cv = std / mean if mean > 0 else np.inf

        if frac < cv_frac_thresh or cv < cv_thresh:
            method = "binary_by_cv_or_frac"
            records.append({
                "marker": marker,
                "marker_method": method,
                "fraction_nonzero": frac,
                "cv": float(cv),
                "p_lower_pct": None,
                "p_upper_pct": None,
                "p1": 0.0,
                "p99": 1.0
            })
            continue

        # determine percentiles
        if frac > 0.30:
            lower_pct, upper_pct = 0.1, 99.9
        elif frac > 0.05:
            lower_pct, upper_pct = 1.0, 99.0
        else:
            lower_pct, upper_pct = 10.0, 90.0

        p1, p99 = np.percentile(nonzero, [lower_pct, upper_pct])
        if (p99 - p1) < min_range:
            p1, p99 = nonzero.min(), nonzero.max()

        method = "robust"

        records.append({
            "marker": marker,
            "marker_method": method,
            "fraction_nonzero": frac,
            "cv": float(cv),
            "p_lower_pct": lower_pct,
            "p_upper_pct": upper_pct,
            "p1": float(p1),
            "p99": float(p99)
        })

    return pd.DataFrame.from_records(records)

