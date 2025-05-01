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
    stats_df    : output of compute_adaptive_marker_stats_from_cohort
    Returns scaled_matrix in [0,1] using cohort-wide scaling strategy.
    """
    scaled = np.zeros_like(norm_matrix)

    for j, m in enumerate(var_names):
        row = stats_df.loc[stats_df["marker"] == m]
        if row.empty:
            continue

        method = row["marker_method"].values[0]
        fallback = row["fallback"].values[0] if "fallback" in row.columns else None
        col = norm_matrix[:, j]

        if method == "all_zero":
            scaled[:, j] = 0.0
            continue

        elif method == "binary_by_cv_or_frac":
            scaled[:, j] = (col > 0).astype(float)
            continue

        elif method == "robust":
            p1 = row["p1"].values[0]
            p99 = row["p99"].values[0]

            with np.errstate(divide='ignore', invalid='ignore'):
                s = (col - p1) / (p99 - p1)
                s = np.clip(s, 0.0, 1.0)
                s[col == 0] = 0.0

            # Fallback transform if robust scaling wiped dynamic range
            if fallback == "log1p":
                transformed = np.log1p(col)
                min_, max_ = np.nanmin(transformed), np.nanmax(transformed)
                s = (transformed - min_) / (max_ - min_) if max_ > min_ else np.zeros_like(transformed)
                s = np.clip(s, 0.0, 1.0)
                s[col == 0] = 0.0

            elif fallback == "sqrt":
                transformed = np.sqrt(col)
                min_, max_ = np.nanmin(transformed), np.nanmax(transformed)
                s = (transformed - min_) / (max_ - min_) if max_ > min_ else np.zeros_like(transformed)
                s = np.clip(s, 0.0, 1.0)
                s[col == 0] = 0.0

            scaled[:, j] = s

    return scaled


def _adaptive_marker_scaling(
    col: np.ndarray,
    raw_area_norm: np.ndarray | None = None,
    min_range: float = 1e-3,
    cv_thresh: float = 0.1,
    cv_frac_thresh: float = 0.10,
    wipeout_frac_thresh: float = 0.95,
    dynrange_thresh: float = 0.1,
) -> tuple[np.ndarray, dict]:
    """
    Adaptive scaling with optional log1p fallback if robust scaling wipes signal.

    Parameters:
        col              : per-cell normalized marker values
        raw_area_norm    : raw / area for fallback dynamic range check
        min_range        : min range for robust scaling
        cv_thresh        : CV threshold for binarization
        cv_frac_thresh   : Fraction nonzero threshold for binarization
        wipeout_frac_thresh : trigger log1p fallback if this % of scaled = 0
        dynrange_thresh  : min dynamic range in raw_area_norm to justify fallback

    Returns:
        scaled_col (np.ndarray) : scaled values in [0,1]
        stats     (dict)        : method + stats used
    """
    nonzero = col[col > 0]
    n = len(col)
    if nonzero.size == 0:
        return np.zeros_like(col), {
            "marker_method": "all_zero",
            "fraction_nonzero": 0.0
        }

    frac = nonzero.size / n
    mean = nonzero.mean()
    std = nonzero.std()
    cv = std / mean if mean > 0 else np.inf

    # === Binary fallback (low signal or sparse) ===
    if frac < cv_frac_thresh or cv < cv_thresh:
        scaled = (col > 0).astype(float)
        return scaled, {
            "marker_method": "binary_by_cv_or_frac",
            "fraction_nonzero": frac,
            "cv": float(cv)
        }

    # === Robust scaling ===
    if frac > 0.30:
        lower_pct, upper_pct = 0.1, 99.9
    elif frac > 0.05:
        lower_pct, upper_pct = 1.0, 99.0
    else:
        lower_pct, upper_pct = 10.0, 90.0

    p1, p99 = np.percentile(nonzero, [lower_pct, upper_pct])
    if (p99 - p1) < min_range:
        p1, p99 = nonzero.min(), nonzero.max()

    with np.errstate(divide='ignore', invalid='ignore'):
        scaled = (col - p1) / (p99 - p1) if (p99 - p1) >= min_range else np.zeros_like(col)

    scaled = np.clip(scaled, 0.0, 1.0)
    scaled[col == 0] = 0.0

    # === Check if scaling wiped out signal ===
    wipeout = np.mean(scaled == 0.0) > wipeout_frac_thresh
    fallback_applied = False

    if wipeout and raw_area_norm is not None:
        raw_nonzero = raw_area_norm[raw_area_norm > 0]
        if raw_nonzero.size > 0:
            dyn_range = np.nanpercentile(raw_nonzero, 99.5) - np.nanpercentile(raw_nonzero, 0.5)
            if dyn_range > dynrange_thresh:
                # Try log1p fallback
                fallback_applied = True
                log_scaled = np.log1p(col)
                log_scaled = log_scaled / np.nanmax(log_scaled) if np.nanmax(log_scaled) > 0 else log_scaled
                log_scaled[col == 0] = 0.0
                scaled = log_scaled

    stats = {
        "marker_method": "robust_fallback_log1p" if fallback_applied else "robust",
        "fraction_nonzero": frac,
        "cv": float(cv),
        "p_lower_pct": lower_pct,
        "p_upper_pct": upper_pct,
        "p1": float(p1),
        "p99": float(p99),
        "wipeout_triggered": wipeout,
        "fallback_applied": fallback_applied,
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
    Cohort-wide marker-wise scaling strategy with adaptive fallback based on sparsity, CV,
    and dynamic range checks. If robust scaling wipes out >95% of signal, fallback to log1p or sqrt.

    Returns
    -------
    pd.DataFrame with columns:
      marker, marker_method, fraction_nonzero, cv, p1, p99, p_lower_pct, p_upper_pct, fallback
    """
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
                "p99": 0.0,
                "fallback": None
            })
            continue

        frac = nonzero.size / n
        mean = nonzero.mean()
        std = nonzero.std()
        cv = std / mean if mean > 0 else np.inf

        if frac < cv_frac_thresh or cv < cv_thresh:
            records.append({
                "marker": marker,
                "marker_method": "binary_by_cv_or_frac",
                "fraction_nonzero": frac,
                "cv": float(cv),
                "p_lower_pct": None,
                "p_upper_pct": None,
                "p1": 0.0,
                "p99": 1.0,
                "fallback": None
            })
            continue

        if frac > 0.30:
            lower_pct, upper_pct = 0.1, 99.9
        elif frac > 0.05:
            lower_pct, upper_pct = 1.0, 99.0
        else:
            lower_pct, upper_pct = 10.0, 90.0

        p1, p99 = np.percentile(nonzero, [lower_pct, upper_pct])
        fallback = None
        if (p99 - p1) < min_range:
            fallback = "minmax"
            p1, p99 = nonzero.min(), nonzero.max()
            if (p99 - p1) < min_range:
                if np.isfinite(nonzero).all():
                    fallback = "log1p"
                    nonzero_log = np.log1p(nonzero)
                    p1, p99 = np.percentile(nonzero_log, [lower_pct, upper_pct])
                    if (p99 - p1) < min_range:
                        fallback = "sqrt"
                        nonzero_sqrt = np.sqrt(nonzero)
                        p1, p99 = np.percentile(nonzero_sqrt, [lower_pct, upper_pct])

        records.append({
            "marker": marker,
            "marker_method": "robust",
            "fraction_nonzero": frac,
            "cv": float(cv),
            "p_lower_pct": lower_pct,
            "p_upper_pct": upper_pct,
            "p1": float(p1),
            "p99": float(p99),
            "fallback": fallback
        })

    return pd.DataFrame.from_records(records)


