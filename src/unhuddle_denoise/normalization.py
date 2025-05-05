import os
import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

def _per_cell_normalize(
    matrix: np.ndarray,
    var_names: List[str],
    sensor_markers: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize each cell either by total intensity or by sum over sensor_markers.

    Parameters
    ----------
    matrix : np.ndarray (n_cells × n_markers)
        Raw intensity values.
    var_names : List[str]
        Names of each column in `matrix`, length == n_markers.
    sensor_markers : Optional[List[str]]
        Subset of `var_names` to use for per-cell normalization.
        If None or empty after filtering, will fall back to total intensity.

    Returns
    -------
    norm_matrix : np.ndarray
        Same shape as `matrix`, each row divided by that cell's total.
    cell_total : np.ndarray
        1D array of length n_cells giving the denominator used for each row.
    """
    n_cells, n_markers = matrix.shape
    # Map marker → index once
    name_to_idx = {name: idx for idx, name in enumerate(var_names)}

    # Determine which indices to sum over
    if sensor_markers:
        valid = [m for m in sensor_markers if m in name_to_idx]
        invalid = set(sensor_markers) - set(valid)
        if invalid:
            logger.warning(
                "The following sensor_markers are not in var_names and will be ignored: %s",
                sorted(invalid)
            )
        idxs = [name_to_idx[m] for m in valid]
        if not idxs:
            logger.warning("No valid sensor_markers found; falling back to total intensity")
            idxs = list(range(n_markers))
    else:
        idxs = list(range(n_markers))

    # Compute cell-wise total
    cell_total = matrix[:, idxs].sum(axis=1)
    median_total = np.median(cell_total)
    logger.debug("Per-cell totals: median=%g, min=%g, max=%g",
                 median_total, cell_total.min(), cell_total.max())

    # Avoid division-by-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_matrix = matrix / cell_total[:, None]
        # Where total was zero, force zeros rather than NaN/inf
        zeros = (cell_total == 0)
        if zeros.any():
            norm_matrix[zeros, :] = 0.0
            logger.debug("Found %d cells with zero total; set those rows to 0", zeros.sum())

    return norm_matrix, cell_total


def apply_cohort_scaling(
    norm_matrix: np.ndarray,
    var_names: List[str],
    stats_df: pd.DataFrame,
    zero_frac_thresh: float = 0.6,
    log_fallback_enabled: bool = True
) -> np.ndarray:
    """
    Scale each column of `norm_matrix` to [0,1] per the cohort stats in `stats_df`.
    Supports 'binary_by_cv_or_frac', 'robust', and any 'fallback_*' methods.
    If >zero_frac_thresh of cells collapse to zero under robust scaling, optionally
    fallback to log1p and update `stats_df` accordingly.
    """
    n_cells, n_markers = norm_matrix.shape
    scaled = np.zeros_like(norm_matrix)

    # Build a name → row-index map just once
    row_idx_map = {
        row.marker: idx
        for idx, row in stats_df.reset_index(drop=True).iterrows()
    }

    for j, marker in enumerate(var_names):
        if marker not in row_idx_map:
            logger.debug(f"Skipping '{marker}': no stats available")
            continue

        row_idx = row_idx_map[marker]
        meta    = stats_df.iloc[row_idx]
        method  = str(meta.marker_method)
        p1, p99 = float(meta.p1), float(meta.p99)
        col     = norm_matrix[:, j]

        # 1) All-zero column
        nonzero_mask = col != 0
        if not nonzero_mask.any():
            logger.info(f"[{marker}] All-zero column — filling with 0s")
            stats_df.at[row_idx, "marker_method"] = "all_zero"
            stats_df.at[row_idx, "p1"]            = 0.0
            stats_df.at[row_idx, "p99"]           = 0.0
            stats_df.at[row_idx, "fallback"]      = "all_zero"
            continue

        # 2) Binary scaling and all zero
        if method in {"binary_by_cv_or_frac", "binary_by_frac"}:
            EPS = 1e-4
            scaled[:, j] = (col > EPS).astype(float)
            continue
        elif method == "all_zero":
            scaled[:, j] = 0.0
            continue

        # 3) Robust or any fallback_* method
        if (method == "robust" or method.startswith("fallback")) and (p99 > p1):
            # base scaling with cohort p1/p99
            denom = p99 - p1
            sc = (col - p1) / denom
            sc = np.clip(sc, 0.0, 1.0)
            sc[col == 0] = 0.0

            # 3a) optional log1p fallback for original 'robust' only
            if method == "robust" and log_fallback_enabled:
                zero_frac = float((sc < 1e-5).mean())
                if zero_frac > zero_frac_thresh:
                    log_col = np.zeros_like(col)
                    log_col[nonzero_mask] = np.log1p(col[nonzero_mask])
                    good = log_col > 0
                    if np.any(good):
                        pmin, pmax = np.nanpercentile(log_col[good], [1, 99])
                        if pmax > pmin:
                            sc = (log_col - pmin) / (pmax - pmin)
                            sc = np.clip(sc, 0.0, 1.0)
                            sc[col == 0] = 0.0
                            stats_df.at[row_idx, "marker_method"] = "fallback_log1p"
                            stats_df.at[row_idx, "p1"]            = pmin
                            stats_df.at[row_idx, "p99"]           = pmax
                            stats_df.at[row_idx, "fallback"]      = "log1p"
                            logger.info(
                                f"[{marker}] log1p fallback applied (zero_frac={zero_frac:.2f})"
                            )
            # assign the scaled array
            scaled[:, j] = sc
            continue

        # 4) Anything else → zeros (unhandled method)
        logger.warning(f"[{marker}] Unrecognized method '{method}'; setting column to 0s")
        scaled[:, j] = 0.0

    return scaled




import numpy as np
import pandas as pd
from typing import List

def compute_adaptive_marker_stats_from_cohort(
        all_matrices: List[np.ndarray],
        var_names: List[str],
        sample_max_cells: int = 100_000,
        min_range: float = 1e-3,
        cv_frac_thresh: float = 0.01,
) -> pd.DataFrame:
    """
    Build cohort‐wide scaling stats for each marker with adaptive fallbacks.

    Steps per marker:
      1. Filter to finite values, optionally downsample across all FOVs.
      2. Compute fraction_nonzero.
      3. If too sparse → binary scaling. (cv_frac_thresh)
      4. Else: compute robust percentiles at adaptive cutoffs.
      5. If (p99-p1)<min_range → try: minmax, log1p, sqrt, in that order.
      *. cutoff for spread of non-zero values per marker is deprecated
    """
    X = np.vstack(all_matrices)
    if X.shape[0] > sample_max_cells:
        idx = np.random.choice(X.shape[0], sample_max_cells, replace=False)
        X = X[idx]

    records = []
    for j, marker in enumerate(var_names):
        col = X[:, j]
        finite = col[np.isfinite(col)]
        n_total = finite.size

        if n_total == 0:
            records.append({
                "marker": marker,
                "marker_method": "all_zero",
                "fraction_nonzero": 0.0,
                "cv": None,
                "p_lower_pct": None,
                "p_upper_pct": None,
                "p1": 0.0,
                "p99": 0.0,
                "fallback": "all_zero"
            })
            continue

        nonzero = finite[finite > 0]
        frac = nonzero.size / n_total
        mean = nonzero.mean() if nonzero.size else 0.0
        std = nonzero.std() if nonzero.size else 0.0
        cv = (std / mean) if mean > 0 else np.inf

        # 1) binary fallback by sparsity only
        if frac < cv_frac_thresh:
            records.append({
                "marker": marker,
                "marker_method": "binary_by_frac",
                "fraction_nonzero": frac,
                "cv": float(cv),
                "p_lower_pct": None,
                "p_upper_pct": None,
                "p1": 0.0,
                "p99": 1.0,
                "fallback": "binary"
            })
            continue

        # 2) choose robust percentile strategy
        if frac > 0.30:
            low_pct, high_pct = 0.1, 99.9
        elif frac > 0.05:
            low_pct, high_pct = 1.0, 99.0
        else:
            low_pct, high_pct = 10.0, 90.0

        p1, p99 = np.nanpercentile(nonzero, [low_pct, high_pct])
        method = "robust"
        fallback = None

        # 3) fallback if range too narrow
        if (p99 - p1) < min_range:
            for fb_name, transform in [
                ("minmax", lambda x: x),
                ("log1p", np.log1p),
                ("sqrt", np.sqrt),
            ]:
                t = transform(nonzero)
                if fb_name == "minmax":
                    lo, hi = t.min(), t.max()
                else:
                    lo, hi = np.nanpercentile(t, [1, 99])
                if (hi - lo) >= min_range:
                    method = f"fallback_{fb_name}"
                    fallback = fb_name
                    p1, p99 = float(lo), float(hi)
                    break

        records.append({
            "marker": marker,
            "marker_method": method,
            "fraction_nonzero": frac,
            "cv": float(cv),  # retained for diagnostic value
            "p_lower_pct": low_pct,
            "p_upper_pct": high_pct,
            "p1": float(p1),
            "p99": float(p99),
            "fallback": fallback
        })

    return pd.DataFrame.from_records(records)




