#!/usr/bin/env python3
import os
import glob
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def load_all_fovs(morph_dir, protein_dir, morph_col):
    """Load and join all FOVs into one DataFrame."""
    all_dfs = []
    fov_names = []
    for prot_path in sorted(glob.glob(os.path.join(protein_dir, "*.csv"))):
        fov = os.path.splitext(os.path.basename(prot_path))[0]
        morph_path = os.path.join(morph_dir, f"{fov}.csv")
        try:
            m = pd.read_csv(morph_path)
            p = pd.read_csv(prot_path)
            if morph_col not in m.columns:
                raise KeyError(f"'{morph_col}' not in {morph_path}")
            df = m[[morph_col]].join(p, how="inner")
            df["fov"] = fov
            all_dfs.append(df)
            fov_names.append(fov)
        except Exception as e:
            logging.warning(f"Skipping FOV {fov}: {e}")
    if not all_dfs:
        raise RuntimeError("No valid FOVs found!")
    full = pd.concat(all_dfs, ignore_index=True)
    return full, fov_names

def simple_percentile_denoise(df, markers, layer_suffix, percentiles):
    """
    For each marker compute the given percentile of non-zero intensities,
    subtract it (clamped at zero) and return a new DataFrame plus metadata.
    """
    denoised = df.copy()
    metadata = {}
    for marker in markers:
        col = marker + layer_suffix
        vals = denoised[col].values
        nonzero = vals[vals > 0]
        thr_dict = {}
        for p in percentiles:
            thr = np.percentile(nonzero, p) if len(nonzero) > 0 else 0.0
            thr_dict[p] = float(thr)
            new_col = f"{marker}{layer_suffix}_p{p}_denoised"
            denoised[new_col] = np.clip(vals - thr, a_min=0, a_max=None)
        metadata[marker] = thr_dict
    return denoised, metadata

def plot_qc(df, markers, metadata, morph_col, layer_suffix, out_dir):
    """Generate a multi-page PDF and individual PNGs for QC."""
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "simple_denoise_QC.pdf")
    png_dir = os.path.join(out_dir, "PNGs")
    os.makedirs(png_dir, exist_ok=True)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    with PdfPages(pdf_path) as pdf:
        for marker in markers:
            thr_dict = metadata[marker]
            percentiles = sorted(thr_dict.keys())
            n_plots = 2 + len(percentiles)
            fig, axs = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), squeeze=False)[0]

            # scatter + percentile lines
            x = df[morph_col].values
            y = df[marker + layer_suffix].values
            axs[0].scatter(x, y, alpha=0.3, s=5)
            for i, p in enumerate(percentiles):
                axs[0].axhline(thr_dict[p], linestyle='--', label=f"p{p}→{thr_dict[p]:.1f}")
            axs[0].set_title(f"{marker}: raw vs {morph_col}")
            axs[0].set_xlabel(morph_col)
            axs[0].set_ylabel("Intensity")
            axs[0].legend(fontsize="small")

            # residuals for each percentile
            for i, p in enumerate(percentiles):
                resid = y - thr_dict[p]
                ax = axs[i + 1]
                ax.scatter(x, resid, alpha=0.3, s=5, color=colors[i % len(colors)])
                ax.axhline(0, linestyle='--', color='gray')
                ax.set_title(f"{marker}: residuals p{p}")
                ax.set_xlabel(morph_col)
                ax.set_ylabel("Residual")

            plt.tight_layout()
            pdf.savefig(fig)
            png_path = os.path.join(png_dir, f"{marker}_QC.png")
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
    logging.info("QC plots written to %s (+ PNGs in %s)", pdf_path, png_dir)

def write_metadata(metadata, out_dir):
    """Dump thresholds metadata to CSV."""
    rows = []
    for marker, thr_dict in metadata.items():
        row = {"marker": marker}
        row.update({f"p{p}_threshold": thr for p, thr in thr_dict.items()})
        rows.append(row)
    df = pd.DataFrame(rows).set_index("marker")
    path = os.path.join(out_dir, "simple_denoise_metadata.csv")
    df.to_csv(path)
    logging.info("Metadata saved to %s", path)

def update_proteins(denoised, protein_dir, layer_suffix, percentiles):
    """For each FOV, append the new denoised columns to its protein CSV."""
    for fov, grp in denoised.groupby("fov"):
        prot_path = os.path.join(protein_dir, fov + ".csv")
        if not os.path.exists(prot_path):
            logging.warning("Missing protein CSV for %s, skipping", fov)
            continue
        prot = pd.read_csv(prot_path)
        # drop old simple-denoised cols if present
        old = [c for c in prot.columns if c.endswith("_denoised")]
        if old:
            prot.drop(columns=old, inplace=True)
        # pick new cols
        new_cols = [c for c in grp.columns if c.endswith("_denoised")]
        block = grp[new_cols].reset_index(drop=True)
        if len(block) != len(prot):
            logging.warning("Row count mismatch for %s, skipping write", fov)
            continue
        out = pd.concat([prot.reset_index(drop=True), block], axis=1)
        out.to_csv(prot_path, index=False)
        logging.info("Updated %s with simple-denoised columns", prot_path)

def run_percentile_denoise(args, dirs):
    """
    Run simple percentile denoising at a single user-specified percentile p,
    then produce QC figures with binned percentile‐curves at [p/2, p, 2p].
    """
    import os
    import logging
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages


    # 1) Directories
    morph_dir   = dirs["morph"]
    protein_dir = dirs["protein"]
    qc_out_dir  = dirs.get(
        "QC_metadata_denoised",
        os.path.join(dirs["QC"], "metadata_denoise")
    )
    os.makedirs(qc_out_dir, exist_ok=True)

    # 2) Parse single percentile p
    p = float(args.percentile)
    low_p, mid_p, high_p = p/2.0, p, p*2.0
    percentiles = [low_p, mid_p, high_p]

    # 3) Morphology column
    if args.denoise_regress.lower() == "perimeter":
        morph_col = "Perimeter"
    elif args.denoise_regress.lower() == "area":
        morph_col = "Area"
    else:
        raise ValueError("`--denoise_regress` must be 'perimeter' or 'area'")

    # 4) Load and combine all FOVs
    full_df, fovs = load_all_fovs(morph_dir, protein_dir, morph_col)

    # 5) Identify markers
    layer_suffix   = "_ExclusionMembrane_Sum_Intensity"
    intensity_cols = [c for c in full_df.columns if c.endswith(layer_suffix)]
    markers = [c[:-len(layer_suffix)] for c in intensity_cols]

    # 6) Compute global thresholds & denoised columns
    denoised = full_df.copy()
    thresholds = {}
    for marker in markers:
        col = marker + layer_suffix
        vals = denoised[col].values
        nz = vals[vals > 0]
        thr = float(np.percentile(nz, p)) if nz.size else 0.0
        thresholds[marker] = thr
        denoised[f"{marker}{layer_suffix}_denoised"] = np.clip(vals - thr, 0, None)

    # 7) Pre-filter markers (once) and log skipped
    min_morph, min_cells = 10, 50
    valid_markers, failed = [], []
    for marker in markers:
        df_m = full_df[[morph_col, marker + layer_suffix]]
        df_m = df_m[(df_m[marker + layer_suffix] > 0) &
                    (df_m[morph_col] >= min_morph)]
        if len(df_m) >= min_cells:
            valid_markers.append(marker)
        else:
            failed.append(marker)

    if failed:
        logging.warning("Markers skipped (too few cells): %s", ", ".join(failed))

    if not valid_markers:
        logging.error("No markers passed filtering (min_cells=%d). Exiting.", min_cells)
        return

    # 8) Build compiled QC figure
    bin_count = 60
    max_points = 5000  # subsample scatter for speed
    n_rows, n_cols = len(valid_markers), 1 + len(percentiles)
    fig, axs = plt.subplots(n_rows, n_cols,
                            figsize=(5 * n_cols, 4 * n_rows),
                            squeeze=False)

    colors = ['red', 'orange', 'green']
    for i, marker in enumerate(valid_markers):
        col = marker + layer_suffix
        df_m = full_df[[morph_col, col]]
        df_m = df_m[(df_m[col] > 0) & (df_m[morph_col] >= min_morph)]

        # binning
        df_m['bin'] = pd.cut(df_m[morph_col], bins=bin_count, labels=False)
        centers = df_m.groupby('bin')[morph_col].mean()
        curves = {
            pv: df_m.groupby('bin')[col]
            .apply(lambda z: np.percentile(z, pv))
            for pv in percentiles
        }

        x_full = df_m[morph_col].values
        y_full = df_m[col].values
        # subsample indices
        if len(x_full) > max_points:
            idx = np.random.choice(len(x_full), max_points, replace=False)
            x, y = x_full[idx], y_full[idx]
        else:
            x, y = x_full, y_full

        # left: scatter + curves
        ax0 = axs[i][0]
        ax0.scatter(x, y, alpha=0.3, s=5, color='blue')
        for j, pv in enumerate(percentiles):
            ax0.plot(centers, curves[pv],
                     linestyle='--', alpha=0.8,
                     color=colors[j], label=f"p{pv:.1f}")
        if i == 0:
            ax0.set_title("Intensity vs " + morph_col)
        if i == n_rows - 1:
            ax0.set_xlabel(morph_col)
        ax0.set_ylabel(marker)
        ax0.legend(fontsize='x-small')

        # residual panels
        for j, pv in enumerate(percentiles):
            ax = axs[i][j + 1]
            resid = y_full - np.interp(x_full, centers, curves[pv])
            # subsample residual scatter too
            if len(x_full) > max_points:
                ax.scatter(x, resid[idx], alpha=0.3, s=5, color=colors[j])
            else:
                ax.scatter(x, resid, alpha=0.3, s=5, color=colors[j])
            ax.axhline(0, linestyle='--', color='gray')
            if i == 0:
                ax.set_title(f"Residuals p{pv:.1f}")
            if i == n_rows - 1:
                ax.set_xlabel(morph_col)

    plt.tight_layout()
    out_path = os.path.join(qc_out_dir, "compiled_denoise_QC.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info("Saved compiled QC to %s", out_path)

    # 9) Metadata CSV & 10) update proteins
    write_metadata({m: {"thr": t} for m, t in thresholds.items()}, qc_out_dir)
    update_proteins(denoised, protein_dir, layer_suffix, [p])

    logging.info("✅ Percentile denoising + compiled QC complete.")











