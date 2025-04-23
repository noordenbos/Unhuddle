import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from PIL import Image
from scipy.spatial import cKDTree
from skimage.transform import resize
from tqdm import tqdm

logger = logging.getLogger(__name__)


def create_directories(output_base_path):
    """
    Create standard output directories under a given base path.

    Returns:
        qc_output_dir, density_dir, segmentation_dir, storyboard_dir
    """
    qc = os.path.join(output_base_path, "QC")
    dens = os.path.join(qc, "density_maps")
    seg = os.path.join(qc, "segmentation")
    sb = os.path.join(qc, "storyboards")
    for d in (qc, dens, seg, sb):
        os.makedirs(d, exist_ok=True)
    return qc, dens, seg, sb


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
    qc_dir,
    density_dir,
    window_size=50,
    stride=10,
    density_threshold=550,
    region_threshold=0.8,
    density_scale=(0,800)
):
    """
    Spatial density QC per FOV with progress bar.
    """
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
        plt.figure(figsize=(6,6)); plt.imshow(dm, cmap='hot', vmin=density_scale[0], vmax=density_scale[1]); plt.axis('off')
        plt.savefig(os.path.join(density_dir, f"{fov}.png"), dpi=200, bbox_inches='tight'); plt.close()
        # determine region cells
        m = dm > density_threshold
        if m.shape != seg.shape:
            m = resize(m.astype(float), seg.shape) > 0.5
        cnt, rg = defaultdict(int), defaultdict(int)
        for y in range(seg.shape[0]):
            for x in range(seg.shape[1]):
                lab = seg[y,x]
                if lab:
                    cnt[lab] += 1
                    if m[y,x]: rg[lab] += 1
        keep = {f"{fov}_{lab}" for lab, c in rg.items() if c/cnt[lab] > region_threshold}
        region_map[fov] = keep
        mask = adata.obs['fov']==fov
        adata.obs.loc[mask, 'QC_filter_low_quality_region'] = adata.obs.loc[mask].index.isin(keep)
    return adata, region_map


from scipy.spatial import cKDTree
import numpy as np
import logging

logger = logging.getLogger(__name__)


def perform_dr_filtering(adata, radius, good_frac=0.75):
    """
    Generic DR-based neighborhood filtering.
    """
    # Find dimensionality reduction key, excluding spatial
    dr_key = next((k for k in adata.obsm if k.startswith('X_') and k != 'X_spatial'), None)

    if dr_key is None:
        print("No DR embedding found in .obsm (e.g., X_tsne, X_umap). Skipping DR filtering. Consider adding Dimension reduction coordinates")
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


def generate_segmentation_images(adata, fovs, density_dir, segmentation_dir, region_map):
    os.makedirs(segmentation_dir, exist_ok=True)
    for fov in tqdm(fovs, desc="🎨 Segmentation QC", unit="FOV"):
        seg = adata.uns.get('spatial',{}).get(fov,{}).get('segmentation')
        if seg is None: continue
        H,W=seg.shape
        low = set(adata.obs.query("fov==@fov and QC_low_intensity_filter").index)
        reg = region_map.get(fov, set())
        drf = set(adata.obs.query("fov==@fov and QC_dr_based_filter").index)
        img = np.zeros((H, W, 3), dtype=np.uint8)  # RGB = (0, 0, 0)
        for y in range(H):
            for x in range(W):
                lab=seg[y,x]
                if not lab: continue
                key=f"{fov}_{lab}"
                if key in low: img[y,x]=(255,0,0)
                elif key in reg: img[y,x]=(0,255,0)
                elif key in drf: img[y,x]=(255,255,0)
                else: img[y, x] = (210, 210, 210)
        plt.imsave(os.path.join(segmentation_dir,f"{fov}.png"), img)
    logger.info("Segmentation overlays done")


def create_storyboard(paths, out, cols=10):
    if not paths: return
    imgs=[Image.open(p) for p in paths]
    w,h=min(i.width for i in imgs), min(i.height for i in imgs)
    board=Image.new('RGB',(cols*w, ((len(imgs)+cols-1)//cols)*h),(255,255,255))
    for i,im in enumerate(imgs): board.paste(im.resize((w,h)),((i%cols)*w,(i//cols)*h))
    board.save(out)


def generate_storyboards(qc_dir, dens_dir, seg_dir, sb_dir, fovs):
    segs=[os.path.join(seg_dir,f"{f}.png") for f in fovs]
    dens=[os.path.join(dens_dir,f"{f}.png") for f in fovs]
    if segs: create_storyboard(segs,os.path.join(sb_dir,'segmentation_storyboard.png'))
    if dens: create_storyboard(dens,os.path.join(sb_dir,'density_storyboard.png'))


import os
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def generate_dr_plot(adata, qc_dir):
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

    os.makedirs(qc_dir, exist_ok=True)
    out_path = os.path.join(qc_dir, 'dr_qc.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"🖼️ DR QC plot saved to: {out_path}")


import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def generate_summary_tables(adata, qc_dir):
    if 'filtering_status' not in adata.obs.columns:
        logger.warning("⚠️ 'filtering_status' missing in AnnData. Skipping summary table generation.")
        return

    os.makedirs(qc_dir, exist_ok=True)

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

    overall_path = os.path.join(qc_dir, 'overall_stats.csv')
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

    per_fov_path = os.path.join(qc_dir, 'per_fov_stats.csv')
    pd.DataFrame(fov_rows).to_csv(per_fov_path, index=False)
    logger.info(f"📄 Wrote per-FOV filtering stats to: {per_fov_path}")

def print_success_guide(output_base_path):
    """
    Print user-friendly summary of QC outputs.
    """
    qc_dir, density_dir, seg_dir, sb_dir = create_directories(output_base_path)
    print("🎉 QC pipeline completed successfully! Here are your visual and numeric outputs:")
    print(f" - Density maps:           {density_dir}")
    print(f" - Segmentation overlays:  {seg_dir}")
    print(f" - Storyboards:            {sb_dir}")

    # Print conditionally if DR filtering was performed
    dr_plot = os.path.join(qc_dir, 'dr_qc.png')
    if os.path.exists(dr_plot):
        print(f" - Dimension Reduction QC plot:             {dr_plot}")

    overall_stats = os.path.join(qc_dir, 'overall_stats.csv')
    if os.path.exists(overall_stats):
        print(f" - Overall filtering stats CSV:      {overall_stats}")

    per_fov_stats = os.path.join(qc_dir, 'per_fov_stats.csv')
    if os.path.exists(per_fov_stats):
        print(f" - Per-FOV filtering stats CSV:      {per_fov_stats}")
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_intensity_distribution(adata, qc_dir, low_intensity_threshold):
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
    os.makedirs(qc_dir, exist_ok=True)
    out_path = os.path.join(qc_dir, "total_intensity_distribution.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

def generate_cohort_normalization_qc(
    adata,
    var_names=None,
    layer_weighted="sum_unhuddle_denoised",
    layer_area="sum_original",
    layer_raw="sum_unhuddle",
    output_dir="qc_dir"
):
    import os
    import numpy as np
    import pandas as pd
    import logging

    logger = logging.getLogger(__name__)

    from .qc_plot_normalization_comparison import (
        plot_sensor_marker_counts,
        plot_intensity_compression,
        plot_marker_correlation_heatmaps,
        plot_marker_distribution_storyboard
    )

    os.makedirs(output_dir, exist_ok=True)
    if var_names is None:
        var_names = adata.var_names.tolist()
    marker_subset = var_names

    logger.info("📊 Generating cohort-level normalization QC")

    # ── Check layer availability ────────────────────────────────────────────────
    if layer_weighted not in adata.layers:
        logger.info(f"Layer '{layer_weighted}' not found in adata. Falling back to '{layer_raw}' for normalization. Consider flag --use denoise.")
        layer_weighted = layer_raw  # fallback to raw if denoised not available

    if layer_area not in adata.layers:
        raise ValueError(f"Required area normalization layer '{layer_area}' not found in adata.")

    # ── Extract layers as DataFrames ────────────────────────────────────────────
    mat_raw = adata.layers[layer_raw]
    mat_weighted = adata.layers[layer_weighted]
    if "Area" not in adata.obs.columns:
        raise KeyError("❌ 'Area' column is missing from adata.obs. This is required for normalization QC.")

    area_array = np.asarray(adata.obs["Area"])
    mat_area = adata.layers[layer_area] / np.clip(area_array[:, None], 1e-5, None)
    raw_df = pd.DataFrame(mat_raw, columns=var_names)
    weighted_df = pd.DataFrame(mat_weighted, columns=var_names)
    area_df = pd.DataFrame(mat_area, columns=var_names)

    # ── Compute marker counts per cell ──────────────────────────────────────────
    marker_counts = (mat_weighted != 0).sum(axis=1)

    # ── Plot 1: Sensor marker counts ────────────────────────────────────────────
    plot_sensor_marker_counts(
        marker_counts=marker_counts,
        output_path=os.path.join(output_dir, "cohort_sensor_marker_counts.png")
    )

    # ── Plot 2: Intensity compression ───────────────────────────────────────────
    plot_intensity_compression(
        raw_df=raw_df,
        norm_df_weighted=weighted_df,
        norm_df_area=area_df,
        marker_subset=marker_subset,
        output_path=os.path.join(output_dir, "cohort_intensity_compression.png")
    )

    # ── Plot 3: Correlation heatmaps ────────────────────────────────────────────
    plot_marker_correlation_heatmaps(
        raw_df=raw_df,
        norm_df_weighted=weighted_df,
        norm_df_area=area_df,
        marker_subset=marker_subset,
        output_path=os.path.join(output_dir, "cohort_marker_correlation.png")
    )

    # ── Plot 4: Marker distribution storyboard ──────────────────────────────────
    plot_marker_distribution_storyboard(
        raw_df=raw_df,
        norm_df_weighted=weighted_df,
        norm_df_area=area_df,
        marker_subset=marker_subset,
        output_path=os.path.join(output_dir, "cohort_marker_distributions_storyboard.png")
    )

    logger.info("✅ Cohort-level normalization QC complete")


def run_qc_from_memory(args, adata):
    qc_dir, dens, seg, sb = create_directories(args.output_base_path)
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
    plot_intensity_distribution(adata, qc_dir, args.low_intensity_threshold)
    adata.obs['QC_low_intensity_filter'] = tot < args.low_intensity_threshold
    low_count = int(adata.obs['QC_low_intensity_filter'].sum())

    # ── 2. Density-Based Region Filter ────────────────────────────────────────────
    logger.debug('🔍 Density QC')
    adata, region_map = perform_density_filtering(
        adata, qc_dir, dens,
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
    generate_segmentation_images(adata, list(region_map), dens, seg, region_map)

    # ── 6. Storyboards ────────────────────────────────────────────────────────────
    logger.info('📚 Storyboards')
    generate_storyboards(qc_dir, dens, seg, sb, list(region_map))
    # ── 6.5 Cohort-Level Normalization QC ───────────────────────────────────────────
    logger.info('📊 Cohort-level normalization QC')
    generate_cohort_normalization_qc(
        adata,
        var_names=adata.var_names.tolist(),
        output_dir=qc_dir
    )

    # ── 7. DR Plot ────────────────────────────────────────────────────────────────
    logger.info('📈 Filtering results in Dimension Reduction plot')
    generate_dr_plot(adata, qc_dir)

    # ── 8. Summary Tables ─────────────────────────────────────────────────────────
    logger.info('📊 Summaries')
    generate_summary_tables(adata, qc_dir)

    # ── 9. Final Keep Flag + Save ─────────────────────────────────────────────────
    logger.info('💾 Save flagged full AnnData (no cells removed)')
    keep = ~(adata.obs.get('QC_low_intensity_filter', False) | adata.obs.get('QC_filter_low_quality_region', False))
    adata.obs['QC_final_keep'] = keep.astype(bool)

    kept_count = int(keep.sum())
    logger.info(f'🔧 Final cells flagged: retained = {kept_count} / {adata.n_obs}')

    out_path = os.path.join(args.output_base_path, 'adata_objects', 'adata1.h5ad')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    adata.write_h5ad(out_path)
    logger.info(f'💾 Saved full QC-flagged AnnData: {out_path}')
    print(f'✅ Saved full QC-flagged AnnData: {out_path}')

    # ── 10. Final Print Summary ───────────────────────────────────────────────────
    del adata
    print_success_guide(args.output_base_path)


