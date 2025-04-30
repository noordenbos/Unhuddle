import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from PIL import Image
from skimage.transform import resize
from tqdm import tqdm
from scipy.spatial import cKDTree
import logging


logger = logging.getLogger(__name__)




def compute_density_map(mask, cells, window_size, stride):
    h, w = mask.shape
    dm = np.zeros((h, w), float)
    ids = set(int(c.split('_')[-1]) for c in cells)
    if not ids.intersection(np.unique(mask)):
        return dm
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            block = mask[y:y+window_size, x:x+window_size]
            cnt = len(set(np.unique(block)).intersection(ids))
            if cnt:
                dm[y:y+window_size, x:x+window_size] += cnt
    return dm


def perform_density_filtering(
    adata,
    dirs,
    window_size=50,
    stride=10,
    density_threshold=550,
    region_threshold=0.8,
    density_scale=(0,800)
):
    """
    Spatial density QC per FOV with progress bar.
    """
    from skimage.transform import resize
    import matplotlib.pyplot as plt

    density_dir = dirs["QC_density"]

    adata.obs['QC_filter_low_quality_region'] = False
    low = set(adata.obs.index[adata.obs['QC_low_intensity_filter']])
    fovs = sorted(adata.obs['fov'].unique())
    region_map = {}

    for fov in tqdm(fovs, desc="🔍 Density QC", unit="FOV"):
        seg = adata.uns.get('spatial', {}).get(fov, {}).get('segmentation')
        if seg is None:
            logger.warning(f"Skipping {fov}, no segmentation.")
            continue
        filtered = [c for c in low if c.startswith(f"{fov}_")]
        dm = compute_density_map(seg, filtered, window_size, stride)
        # save
        plt.figure(figsize=(6, 6))
        plt.imshow(dm, cmap='hot', vmin=density_scale[0], vmax=density_scale[1])
        plt.axis('off')
        plt.savefig(os.path.join(density_dir, f"{fov}.png"), dpi=200, bbox_inches='tight')
        plt.close()
        # determine region cells
        m = dm > density_threshold
        if m.shape != seg.shape:
            m = resize(m.astype(float), seg.shape) > 0.5
        cnt, rg = defaultdict(int), defaultdict(int)
        for y in range(seg.shape[0]):
            for x in range(seg.shape[1]):
                lab = seg[y, x]
                if lab:
                    cnt[lab] += 1
                    if m[y, x]:
                        rg[lab] += 1
        keep = {f"{fov}_{lab}" for lab, c in rg.items() if c / cnt[lab] > region_threshold}
        region_map[fov] = keep
        mask = adata.obs['fov'] == fov
        adata.obs.loc[mask, 'QC_filter_low_quality_region'] = adata.obs.loc[mask].index.isin(keep)
    return adata, region_map





def perform_dr_filtering(adata, radius, good_frac=0.75):
    """
    Generic DR-based neighborhood filtering.
    """
    # Find dimensionality reduction key, excluding spatial
    dr_key = next((k for k in adata.obsm if k.startswith('X_') and k != 'X_spatial'), None)

    if dr_key is None:
        logger.info("No DR embedding found in .obsm (e.g., X_tsne, X_umap). Skipping DR filtering. Consider adding Dimension reduction coordinates")
        adata.obs['QC_fraction_filtered'] = np.nan
        adata.obs['QC_dr_based_filter'] = False
        return adata

    logger.info(f"🔧 Using DR embedding: {dr_key}")
    pts = adata.obsm[dr_key]

    if pts.shape[0] != adata.n_obs:
        logger.error("❌ DR coordinate shape does not match number of cells in AnnData.")
        raise ValueError(f"DR shape mismatch: DR coords have {pts.shape[0]}, adata has {adata.n_obs} cells")

    logger.debug(f"DR filter on {pts.shape[0]} cells with radius={radius}")
    good = adata.obs['filtering_status'] == 'Unfiltered'
    tree_all = cKDTree(pts)
    pts_good = pts[good.values]

    if pts_good.size == 0:
        logger.warning("⚠️ No 'Unfiltered' cells found for DR-based filtering.")
        adata.obs['QC_fraction_filtered'] = 1.0
        adata.obs['QC_dr_based_filter'] = False
        return adata

    tree_good = cKDTree(pts_good)

    tot = np.array([len(n) for n in tree_all.query_ball_point(pts, r=radius)])
    ok = np.array([len(n) for n in tree_good.query_ball_point(pts, r=radius)])

    frac = np.zeros_like(tot, dtype=float)
    valid = tot > 0
    frac[valid] = ok[valid] / tot[valid]

    adata.obs['QC_fraction_filtered'] = frac
    flag = frac < good_frac
    adata.obs['QC_dr_based_filter'] = flag

    msg = int(flag.sum())
    logger.info(f"🎯 DR filtering flagged {msg}/{len(frac)} cells (frac < {good_frac})")

    idx = flag & (adata.obs['filtering_status'] == 'Unfiltered')
    adata.obs.loc[idx, 'filtering_status'] = 'bad dr cluster'

    return adata


def generate_segmentation_images(adata, fovs, region_map,dirs):
    import matplotlib.pyplot as plt

    seg_dir = dirs["QC_segmentation"]
    os.makedirs(seg_dir, exist_ok=True)

    for fov in tqdm(fovs, desc="🎨 Segmentation QC", unit="FOV"):
        seg = adata.uns.get('spatial', {}).get(fov, {}).get('segmentation')
        if seg is None:
            continue
        H, W = seg.shape
        low = set(adata.obs.query("fov == @fov and QC_low_intensity_filter").index)
        reg = region_map.get(fov, set())
        drf = set(adata.obs.query("fov == @fov and QC_dr_based_filter").index)

        img = np.zeros((H, W, 3), dtype=np.uint8)
        for y in range(H):
            for x in range(W):
                lab = seg[y, x]
                if not lab:
                    continue
                key = f"{fov}_{lab}"
                if key in low:
                    img[y, x] = (255, 0, 0)
                elif key in reg:
                    img[y, x] = (0, 255, 0)
                elif key in drf:
                    img[y, x] = (255, 255, 0)
                else:
                    img[y, x] = (210, 210, 210)

        plt.imsave(os.path.join(seg_dir, f"{fov}.png"), img)

    logger.info("Segmentation overlays done")





def create_storyboard(paths, out_path, cols=10):
    """
    Create a storyboard image from a list of image paths and save it to out_path.

    Args:
        paths (list): List of image file paths to include in the storyboard.
        out_path (str): Output path for the storyboard PNG.
        cols (int): Number of columns in the storyboard grid.
    """
    if not paths:
        return

    imgs = [Image.open(p) for p in paths]
    w, h = min(img.width for img in imgs), min(img.height for img in imgs)

    board = Image.new('RGB', (cols * w, ((len(imgs) + cols - 1) // cols) * h), (255, 255, 255))
    for i, img in enumerate(imgs):
        board.paste(img.resize((w, h)), ((i % cols) * w, (i // cols) * h))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    board.save(out_path)



def generate_storyboards(dirs, fovs):
    """
    Generate storyboard images for segmentation and density QC maps.

    Args:
        dirs (dict): Dictionary containing paths including:
            - dirs["QC_density"]
            - dirs["QC_segmentation"]
            - dirs["QC_storyboards"]
        fovs (list): List of FOV names.
    """
    seg_dir = dirs["QC_segmentation"]
    dens_dir = dirs["QC_density"]
    sb_dir = dirs["QC_storyboards"]

    seg_paths = [os.path.join(seg_dir, f"{f}.png") for f in fovs]
    dens_paths = [os.path.join(dens_dir, f"{f}.png") for f in fovs]

    seg_out = os.path.join(sb_dir, "segmentation_storyboard.png")
    dens_out = os.path.join(sb_dir, "density_storyboard.png")

    if seg_paths:
        create_storyboard(seg_paths, seg_out)
    if dens_paths:
        create_storyboard(dens_paths, dens_out)


def generate_dr_plot(adata, dirs):
    """
    Generate a scatter plot of cells colored by filtering status in DR space.

    Args:
        adata (AnnData): The annotated data matrix.
        dirs (dict): Dictionary containing paths, including dirs["QC"].
    """
    # Look for any DR embedding other than X_spatial
    dr_key = next((k for k in adata.obsm if k.startswith('X_') and k != 'X_spatial'), None)

    if dr_key is None:
        logger.info("No DR embedding found in .obsm. Skipping DR plot.")
        return

    coords = adata.obsm[dr_key]
    if coords.shape[0] != adata.n_obs:
        logger.error(f"❌ DR coords shape mismatch: {coords.shape[0]} rows vs {adata.n_obs} obs.")
        return

    if 'filtering_status' not in adata.obs.columns:
        logger.warning("⚠️ 'filtering_status' not found in obs. Cannot generate DR plot.")
        return

    status = adata.obs['filtering_status'].values
    colors = np.full(len(coords), 'lightgray', dtype=object)
    colors[status == 'low intensity cell'] = 'red'
    colors[status == 'low quality region'] = 'green'
    colors[status == 'bad dr cluster'] = 'yellow'

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=1, alpha=0.6)

    ax.set_title(f"Filtering Results after Dimension Reduction ({dr_key})")
    ax.set_xlabel(f"{dr_key} 1")
    ax.set_ylabel(f"{dr_key} 2")

    legend_handles = [
        plt.Line2D([], [], marker='o', color='w', markerfacecolor=color, label=label, markersize=10)
        for label, color in [
            ('Low Intensity Cell', 'red'),
            ('Low Quality Region', 'green'),
            ('Bad DR Cluster', 'yellow'),
            ('Unfiltered', 'lightgray')
        ]
    ]
    ax.legend(
        handles=legend_handles,
        title='Filter',
        loc='upper right',
        markerscale=0.5,
        fontsize='x-small',
        title_fontsize='small'
    )

    out_path = os.path.join(dirs["QC"], 'dr_qc.png')
    os.makedirs(dirs["QC"], exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"🖼️ DR QC plot saved to: {out_path}")


def generate_summary_tables(adata, dirs):
    """
    Generate summary CSVs of filtering results at overall and per-FOV level.

    Args:
        adata (AnnData): The annotated data matrix.
        dirs (dict): Dictionary with paths, must include dirs["QC"].
    """

    if 'filtering_status' not in adata.obs.columns:
        logger.warning("⚠️ 'filtering_status' missing in AnnData. Skipping summary table generation.")
        return

    status = adata.obs['filtering_status']
    total = len(status)

    steps = [
        ('Total', None),
        ('Low Intensity', 'low intensity cell'),
        ('Low Quality', 'low quality region'),
        ('Bad DR', 'bad dr cluster')
    ]

    rows = [
        {
            'Filter Step': name,
            'Count': (status == val).sum() if val else total,
            'Percent': ((status == val).sum() / total * 100) if val else 100.0
        }
        for name, val in steps
    ]

    overall_path = os.path.join(dirs["QC_filtering"], 'overall_stats.csv')
    pd.DataFrame(rows).to_csv(overall_path, index=False)
    logger.info(f"📄 Wrote overall filtering stats to: {overall_path}")

    # Per-FOV stats
    if 'fov' not in adata.obs.columns:
        logger.warning("⚠️ 'fov' not found in obs. Skipping per-FOV stats.")
        return

    fov_rows = []
    for fov in adata.obs['fov'].unique():
        sub = status[adata.obs['fov'] == fov]
        row = {'FOV': fov}
        for name, val in steps:
            row[name] = (sub == val).sum() if val else len(sub)
        fov_rows.append(row)

    per_fov_path = os.path.join(dirs["QC_filtering"], 'per_fov_stats.csv')
    pd.DataFrame(fov_rows).to_csv(per_fov_path, index=False)
    logger.info(f"📄 Wrote per-FOV filtering stats to: {per_fov_path}")

def print_success_guide(dirs):
    """
    Print user-friendly summary of QC outputs using standard dirs structure.
    """
    print("🎉 QC pipeline completed successfully! Here are your visual and numeric outputs:")

    print(f" - Density maps:           {dirs['QC_density']}")
    print(f" - Segmentation overlays:  {dirs['QC_segmentation']}")
    print(f" - Storyboards:            {dirs['QC_storyboards']}")

    dr_plot = os.path.join(dirs["QC_filtering"], 'dr_qc.png')
    if os.path.exists(dr_plot):
        print(f" - Dimension Reduction QC plot:      {dr_plot}")

    overall_stats = os.path.join(dirs["QC_filtering"], 'overall_stats.csv')
    if os.path.exists(overall_stats):
        print(f" - Overall filtering stats CSV:      {overall_stats}")

    per_fov_stats = os.path.join(dirs["QC_filtering"], 'per_fov_stats.csv')
    if os.path.exists(per_fov_stats):
        print(f" - Per-FOV filtering stats CSV:      {per_fov_stats}")



def plot_intensity_distribution(adata, dirs, low_intensity_threshold):
    """
    Plot total intensity distribution across all cells, with a line at the filtering threshold.
    """
    data = adata.obs["total_intensity"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Zoomed view (0–45)
    bins_a = np.arange(0, 45 + 0.2, 0.2)
    axes[0].hist(data, bins=bins_a, color='skyblue', edgecolor='black')
    axes[0].axvline(low_intensity_threshold, color='red', linestyle='--', linewidth=2)
    axes[0].text(low_intensity_threshold + 0.5, axes[0].get_ylim()[1] * 0.9,
                 f"--low_intensity_threshold = {low_intensity_threshold}",
                 color='red', va='top', ha='left', fontsize=10)
    axes[0].set_title("Summed Intensity (0–45)")

    # Full range view
    axes[1].hist(data, bins=100, color='lightgray', edgecolor='black')
    axes[1].axvline(low_intensity_threshold, color='red', linestyle='--', linewidth=2)
    axes[1].text(low_intensity_threshold + (data.max() * 0.01), axes[1].get_ylim()[1] * 0.9,
                 f"--low_intensity_threshold = {low_intensity_threshold}",
                 color='red', va='top', ha='left', fontsize=10)
    axes[1].set_title("Summed Intensity (Full Range)")

    plt.tight_layout()
    out_path = os.path.join(dirs["QC_filtering"], "total_intensity_distribution.png")
    plt.savefig(out_path, dpi=150)
    plt.close()



def qc_plot_normalization_comparison_from_X_png(
    adata,
    dirs,
    raw_layer="sum_unhuddle_denoised",
    area_key="Area",
    marker_subset=None
):
    """
    Creates PNG plots of normalization comparison: Raw, Raw/Area, Scaled from adata.X.
    Saves individual marker plots and one summary plot to QC_plot subdir of dirs.
    """
    out_dir = dirs.get("QC_plot", os.path.join(dirs["QC"], "normalisation_plots"))
    os.makedirs(out_dir, exist_ok=True)

    raw_mat = adata.layers[raw_layer]
    scaled_mat = adata.X
    area = adata.obs[area_key].values

    if marker_subset is None:
        marker_subset = adata.var_names.tolist()

    marker_idx = [i for i, name in enumerate(adata.var_names) if name in marker_subset]
    area_safe = np.where(area == 0, np.nan, area)

    # Combined plot
    raw_all = raw_mat[:, marker_idx].flatten()
    raw_area_norm_all = (raw_mat[:, marker_idx] / area_safe[:, None]).flatten()
    scaled_all = scaled_mat[:, marker_idx].flatten()
    area_all = np.repeat(area, len(marker_idx))

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].scatter(area_all, raw_all, s=1, alpha=0.2)
    axs[0].set_title("All markers - Raw")
    axs[1].scatter(area_all, raw_area_norm_all, s=1, alpha=0.2)
    axs[1].set_title("All markers - Raw / Area")
    axs[2].scatter(area_all, scaled_all, s=1, alpha=0.2)
    axs[2].set_title("All markers - Scaled")
    for ax in axs:
        ax.set_xlabel("Area")
    axs[0].set_ylabel("Intensity")
    fig.suptitle("Normalization Summary (All Markers)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "norm_comparison_summary.png"), dpi=200)
    plt.close()

    from io import BytesIO
    from PIL import Image

    # Create individual per-marker plots as PIL images
    marker_images = []
    for j in marker_idx:
        name = adata.var_names[j]
        raw = raw_mat[:, j]
        raw_area_norm = raw / area_safe
        scaled = scaled_mat[:, j]

        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        axs[0].scatter(area, raw, s=2, alpha=0.3)
        axs[0].set_title("Raw")
        axs[1].scatter(area, raw_area_norm, s=2, alpha=0.3)
        axs[1].set_title("Raw / Area")
        axs[2].scatter(area, scaled, s=2, alpha=0.3)
        axs[2].set_title("Scaled")

        for ax in axs:
            ax.set_xlabel("Area")
        axs[0].set_ylabel(name)

        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        marker_images.append(img)

    # Create storyboard image (1 row per marker, 3 panels wide)
    if marker_images:
        w, h = marker_images[0].size
        storyboard = Image.new("RGB", (w, h * len(marker_images)), (255, 255, 255))
        for i, img in enumerate(marker_images):
            storyboard.paste(img, (0, i * h))

        storyboard_path = os.path.join(out_dir, "normalization_per marker.png")
        storyboard.save(storyboard_path)
        logger.info(f"🖼️ Per marker storyboard saved to: {storyboard_path}")

    return f"✅ PNG normalization plots saved to {out_dir}"
def extend_dirs_with_qc(dirs):
    """
    Extend an existing dirs dictionary with standardized QC subfolders.
    Assumes dirs["QC"] already exists.
    """
    dirs["QC_density"] = os.path.join(dirs["QC"], "density_maps")
    dirs["QC_segmentation"] = os.path.join(dirs["QC"], "segmentation")
    dirs["QC_storyboards"] = os.path.join(dirs["QC"], "storyboards")
    dirs["QC_filtering"] = os.path.join(dirs["QC"], "filtering")

    for key in ["QC_density", "QC_segmentation", "QC_storyboards", "QC_filtering"]:
        os.makedirs(dirs[key], exist_ok=True)

    return dirs


def run_qc_from_memory(args, adata, dirs):
    dirs = extend_dirs_with_qc(dirs)
    logger.debug('🚀 Running QC pipeline')
    logger.info(f'🔧 Low-intensity threshold = {args.low_intensity_threshold}')
    logger.info(f'🔧 Density window = {args.qc_window_size}, stride = {args.qc_stride}')
    logger.info(f'🔧 Density threshold = {args.qc_density_threshold}, region threshold = {args.qc_region_threshold}')
    logger.info(f'🔧 DR radius = {args.radius_DRfilter}')

    # ── 1. Intensity Filter ───────────────────────────────────────────────────────
    if 'sum_unhuddle' not in adata.layers:
        logger.error("❌ Missing 'sum_unhuddle' layer in AnnData. Cannot compute total intensity.")
        raise ValueError("Missing layer: 'sum_unhuddle'")

    tot = adata.layers['sum_unhuddle'].sum(axis=1)
    adata.obs['total_intensity'] = tot
    plot_intensity_distribution(adata, dirs, args.low_intensity_threshold)
    adata.obs['QC_low_intensity_filter'] = tot < args.low_intensity_threshold
    low_count = int(adata.obs['QC_low_intensity_filter'].sum())

    # ── 2. Density-Based Region Filter ────────────────────────────────────────────
    logger.debug('🔍 Density QC')
    adata, region_map = perform_density_filtering(
        adata, dirs,
        window_size=args.qc_window_size,
        stride=args.qc_stride,
        density_threshold=args.qc_density_threshold,
        region_threshold=args.qc_region_threshold,
        density_scale=tuple(args.qc_plot_density_scale)
    )
    region_count = int(adata.obs['QC_filter_low_quality_region'].sum())
    logger.info(f'🔧 Post-density filter counts: low_intensity={low_count}, low_quality_region={region_count}')

    # ── 3. Init Filtering Status ──────────────────────────────────────────────────
    adata.obs['filtering_status'] = 'Unfiltered'
    adata.obs.loc[adata.obs['QC_filter_low_quality_region'], 'filtering_status'] = 'low quality region'
    adata.obs.loc[adata.obs['QC_low_intensity_filter'], 'filtering_status'] = 'low intensity cell'

    # ── 4. DR-Based Filter ────────────────────────────────────────────────────────
    logger.info('🎲 Dimension Reduction (DR) based filtering')
    adata = perform_dr_filtering(adata, args.radius_DRfilter)
    if 'QC_dr_based_filter' in adata.obs:
        dr_bad = int(adata.obs['QC_dr_based_filter'].sum())
        logger.info(f'🔧 Post-DR filter count: bad_dr_cluster = {dr_bad}')
    else:
        logger.info("DR-based filtering skipped or failed — no 'QC_dr_based_filter' in obs. Consider adding dimension reduction coordinates")

    # ── 5. Segmentation Overlays ──────────────────────────────────────────────────
    logger.info('🎨 Segmentation overlays')
    generate_segmentation_images(adata, list(region_map), region_map,dirs)

    # ── 6. Storyboards ────────────────────────────────────────────────────────────
    logger.info('📚 Storyboards')
    generate_storyboards(dirs, list(region_map))

    # ── 6.5 Cohort-Level Normalization QC ─────────────────────────────────────────
    logger.info('📊 Cohort-level normalization QC')
    qc_plot_normalization_comparison_from_X_png(
        adata=adata,
        dirs=dirs,
        marker_subset=None
    )

    # ── 7. DR Plot ────────────────────────────────────────────────────────────────
    logger.info('📈 Filtering results in Dimension Reduction plot')
    generate_dr_plot(adata, dirs)

    # ── 8. Summary Tables ─────────────────────────────────────────────────────────
    logger.info('📊 Summaries')
    generate_summary_tables(adata, dirs)

    # ── 9. Final Keep Flag + Save ─────────────────────────────────────────────────
    logger.info('💾 Save flagged full AnnData (no cells removed)')
    keep = ~(adata.obs.get('QC_low_intensity_filter', False) | adata.obs.get('QC_filter_low_quality_region', False))
    adata.obs['QC_final_keep'] = keep.astype(bool)

    kept_count = int(keep.sum())
    logger.info(f'🔧 Final cells flagged: retained = {kept_count} / {adata.n_obs}')

    out_path = os.path.join(dirs['adata'], 'adata1.h5ad')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    adata.write_h5ad(out_path)
    logger.info(f'💾 Saved full QC-flagged AnnData: {out_path}')
    print(f'✅ Saved full QC-flagged AnnData: {out_path}')

    # ── 10. Final Print Summary ───────────────────────────────────────────────────
    del adata
    print_success_guide(dirs)




