import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)


def create_directories(output_base_path):
    """
    Create standard output directories under a given base path.

    Returns:
        qc_output_dir, density_dir, segmentation_dir, storyboard_dir
    """
    qc_output_dir = os.path.join(output_base_path, "QC")
    density_dir = os.path.join(qc_output_dir, "density_maps")
    segmentation_dir = os.path.join(qc_output_dir, "segmentation")
    storyboard_dir = os.path.join(qc_output_dir, "storyboards")

    for path in [qc_output_dir, density_dir, segmentation_dir, storyboard_dir]:
        os.makedirs(path, exist_ok=True)

    return qc_output_dir, density_dir, segmentation_dir, storyboard_dir


def low_intensity_filter(adata, threshold, key="total_intensity"):
    if key not in adata.obs.columns:
        raise ValueError(f"'{key}' not found in adata.obs. Compute it first.")
    mask = adata.obs[key] >= threshold
    logging.info(f"📉 Dropping {np.sum(~mask)} cells below intensity threshold of {threshold}")
    return adata[mask].copy()


def compute_density_map(segmentation_mask, filtered_cells, window_size=50, stride=10):
    """
    Compute a density map over a segmentation mask.
    """
    height, width = segmentation_mask.shape
    density_map = np.zeros((height, width), dtype=np.float32)
    filtered_cells_numeric = set(int(cell_id.split("_")[-1]) for cell_id in filtered_cells)
    if not any(cid in np.unique(segmentation_mask) for cid in filtered_cells_numeric):
        return density_map

    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            window = segmentation_mask[y:y + window_size, x:x + window_size]
            unique_filtered = set(np.unique(window)).intersection(filtered_cells_numeric)
            if unique_filtered:
                density_map[y:y + window_size, x:x + window_size] += len(unique_filtered)
    return density_map


def perform_density_filtering(
        adata,
        qc_output_dir,
        density_dir,
        window_size=50,
        stride=10,
        density_threshold=550,
        region_threshold=0.8,
        density_scale=(0, 800)
):
    """
    Loop over each FOV to compute density maps with a progress bar.
    """
    filtered_cells_li = set(adata.obs.index[adata.obs["QC_low_intensity_filter"]])
    fovs = list(set(adata.obs["fov"]))
    region_cells_by_fov = {}
    adata.obs["QC_filter_low_quality_region"] = False

    for fov in tqdm(fovs, desc="🔍 Density QC", unit="FOV"):
        spatial = adata.uns.get("spatial", {}).get(fov, {})
        if "segmentation" not in spatial:
            logger.warning(f"⚠️ Skipping {fov} – no segmentation mask.")
            continue

        seg_mask = spatial["segmentation"]
        fov_filtered = {c for c in filtered_cells_li if c.startswith(f"{fov}_")}
        dm = compute_density_map(seg_mask, fov_filtered, window_size, stride)

        # save density map
        plt.figure(figsize=(10, 10))
        plt.imshow(dm, cmap="hot", vmin=density_scale[0], vmax=density_scale[1])
        plt.axis("off")
        plt.savefig(os.path.join(density_dir, f"{fov}.png"), dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()

        # compute region cells
        mask = dm > density_threshold
        if mask.shape != seg_mask.shape:
            from skimage.transform import resize
            mask = resize(mask.astype(float), seg_mask.shape) > 0.5

        counts, region = defaultdict(int), defaultdict(int)
        H, W = seg_mask.shape
        for y in range(H):
            for x in range(W):
                label = seg_mask[y, x]
                if not label: continue
                counts[label] += 1
                if mask[y, x]:
                    region[label] += 1

        kept = {f"{fov}_{lab}" for lab, ct in region.items() if ct / counts[lab] > region_threshold}
        region_cells_by_fov[fov] = kept
        # Set QC flag for this FOV only on matching cells
        fov_mask = adata.obs["fov"] == fov
        adata.obs.loc[fov_mask, "QC_filter_low_quality_region"] = adata.obs.loc[fov_mask].index.isin(kept)

    return adata, region_cells_by_fov


def perform_tsne_filtering(adata, radius):
    """
    Perform t-SNE based neighborhood filtering matching segmentation logic.
    """
    from scipy.spatial import cKDTree
    # Extract t-SNE coordinates
    coords = adata.obsm.get("X_tsne")
    if coords is None:
        logger.error("❌ Missing X_tsne in obsm; skipping t-SNE filtering.")
        return adata
    x = coords[:, 0]
    y = coords[:, 1]
    logger.debug(f"🧪 t-SNE filtering on {len(x)} cells with radius={radius}")

    # Identify cells surviving intensity and density QC (filtering_status == 'Unfiltered')
    good_mask = adata.obs["filtering_status"] == "Unfiltered"
    num_good = int(good_mask.sum())
    logger.debug(f"🔍 {num_good} cells remain unfiltered and will seed neighborhood counts")

    # Build KD-trees
    pts = np.column_stack((x, y))
    tree_all = cKDTree(pts)
    pts_good = pts[good_mask.values]
    if len(pts_good) == 0:
        logger.warning("⚠️ No unfiltered cells to build KD-tree for t-SNE filtering.")
        adata.obs["QC_fraction_filtered"] = 0.0
        adata.obs["QC_tsne_based_filter"] = False
        return adata
    tree_good = cKDTree(pts_good)
    logger.debug("🌲 KD-trees built (all vs. good cells)")

    # Query neighbor counts
    neighbors_total = tree_all.query_ball_point(pts, r=radius)
    count_total = np.array([len(n) for n in neighbors_total], dtype=int)
    neighbors_good = tree_good.query_ball_point(pts, r=radius)
    count_good = np.array([len(n) for n in neighbors_good], dtype=int)
    logger.debug(f"🔢 Example counts (total, good): {list(zip(count_total[:5], count_good[:5]))}")

    # Compute fraction of good neighbors
    fraction = np.zeros_like(count_total, dtype=float)
    valid = count_total > 0
    fraction[valid] = count_good[valid] / count_total[valid]
    adata.obs["QC_fraction_filtered"] = fraction
    logger.debug(f"📊 QC_fraction_filtered stats: min={fraction.min():.4f}, max={fraction.max():.4f}")

    # Flag as bad if too many neighbors were filtered (>75% filtered => good fraction < 0.25)
    tsne_thresh = 0.75
    flag = fraction < tsne_thresh
    adata.obs["QC_tsne_based_filter"] = flag
    logger.info(
        f"🎯 t-SNE filtering flagged {int(flag.sum())}/{len(flag)} cells (good neighbor fraction < {tsne_thresh})")

    # Update filtering_status
    to_update = flag & (adata.obs["filtering_status"] == "Unfiltered")
    adata.obs.loc[to_update, "filtering_status"] = "bad tsne cluster"
    logger.debug(f"🔄 Updated filtering_status for {int(to_update.sum())} cells")

    return adata
    x = coords[:, 0]
    y = coords[:, 1]
    logger.debug(f"🧪 Performing t-SNE filtering with radius={radius} on {len(x)} cells")

    # Determine which cells have already been filtered
    filtered_mask = adata.obs.get("QC_low_intensity_filter", False) | adata.obs.get("QC_filter_low_quality_region",
                                                                                    False)
    logger.debug(f"🔍 Prefilter mask: {filtered_mask.sum()} cells flagged (intensity or region)")
    x_filtered = x[~filtered_mask]
    y_filtered = y[~filtered_mask]
    logger.debug(f"🌲 Building KD-trees: all_points={len(x)}, filtered_points={len(x_filtered)}")

    tree_all = cKDTree(np.column_stack((x, y)))
    tree_filtered = cKDTree(np.column_stack((x_filtered, y_filtered)))

    pts = np.column_stack((x, y))
    logger.debug("📡 Querying neighbor counts for all cells")
    neighbors_total = tree_all.query_ball_point(pts, r=radius)
    count_total = np.array([len(n) for n in neighbors_total], dtype=int)
    neighbors_filtered = tree_filtered.query_ball_point(pts, r=radius)
    count_filtered = np.array([len(n) for n in neighbors_filtered], dtype=int)
    logger.debug(f"🔢 Sample neighbor counts (first5): total={count_total[:5]}, filtered={count_filtered[:5]}")

    # Compute fraction of filtered neighbors
    fraction = np.zeros_like(count_total, dtype=float)
    valid = count_total > 0
    fraction[valid] = count_filtered[valid] / count_total[valid]
    adata.obs["QC_fraction_filtered"] = fraction
    logger.debug(f"📊 QC_fraction_filtered stats: min={fraction.min():.4f}, max={fraction.max():.4f}")

    # Flag cells above threshold
    tsne_threshold = 0.25
    flag = fraction > tsne_threshold
    adata.obs["QC_tsne_based_filter"] = flag
    logger.info(f"🎯 t-SNE filtering flagged {int(flag.sum())}/{len(flag)} cells at threshold={tsne_threshold}")

    # Update filtering_status for newly flagged cells
    to_update = flag & (adata.obs["filtering_status"] == "Unfiltered")
    adata.obs.loc[to_update, "filtering_status"] = "bad tsne cluster"
    logger.debug(f"🔄 Updated filtering_status for {int(to_update.sum())} cells")

    return adata
    pts = np.asarray(coords)

    # Prefilter mask (use QC flags, not filtering_status)
    mask = adata.obs.get("QC_low_intensity_filter", False) | adata.obs.get("QC_filter_low_quality_region", False)
    num_prefilter = int(mask.sum())
    logger.debug(f"🔍 Prefilter mask sum (intensity or region): {num_prefilter}")(
        f"🧪 t-SNE filtering with BallTree: total={pts.shape[0]} pts, filtered mask sum={num_prefilter}")

    # Build BallTrees
    tree_all = BallTree(pts)
    logger.debug("🌲 Built BallTree for all points")
    if num_prefilter > 0:
        pts_filt = pts[mask.values]
        tree_filt = BallTree(pts_filt)
        logger.debug(f"🌲 Built BallTree for {pts_filt.shape[0]} filtered points")
    else:
        tree_filt = None
        logger.debug("⚠️ No filtered points; skipping filtered BallTree build")

    # Count neighbors within radius
    count_total = tree_all.query_radius(pts, r=radius, count_only=True)
    logger.debug("⏱ Completed total neighbor counts")
    if tree_filt is not None:
        count_filtered = tree_filt.query_radius(pts, r=radius, count_only=True)
        logger.debug("⏱ Completed filtered neighbor counts")
    else:
        count_filtered = np.zeros_like(count_total, dtype=int)

    # Compute fraction safely
    fraction = np.zeros_like(count_total, dtype=float)
    nonzero = count_total > 0
    fraction[nonzero] = count_filtered[nonzero] / count_total[nonzero]
    adata.obs["QC_fraction_filtered"] = fraction
    logger.debug(f"📊 QC_fraction_filtered: min={fraction.min():.4f}, max={fraction.max():.4f}")

    # Flagging using threshold
    thresh = 0.25
    flag = fraction > thresh
    # Check for excessive filtering
    flagged_count = int(flag.sum())
    total_cells = len(fraction)
    flagged_fraction = flagged_count / total_cells if total_cells > 0 else 0
    if flagged_fraction > 0.5:
        raise ValueError(
            f"❌ Too many cells flagged by t-SNE filter ({flagged_count}/{total_cells} = {flagged_fraction:.1%}). "
            f"The radius ({radius}) may be too high."
        )
    adata.obs["QC_tsne_based_filter"] = flag
    logger.info(f"🎯 t-SNE filtering flagged {int(flag.sum())} cells at threshold={thresh}")

    # Update filtering_status
    to_update = flag & (adata.obs["filtering_status"] == "Unfiltered")
    adata.obs.loc[to_update, "filtering_status"] = "bad tsne cluster"
    logger.debug(f"🔄 Updated status for {int(to_update.sum())} cells")

    return adata
    pts = np.asarray(coords)
    # Prefilter mask
    mask = adata.obs["filtering_status"] != "Unfiltered"
    logger.debug(f"🧪 t-SNE filtering with BallTree: total={pts.shape[0]} pts, filtered mask sum={mask.sum()}")

    # Build trees
    tree_all = BallTree(pts)
    logger.debug("🌲 Built BallTree for all points")
    pts_filt = pts[mask.values]
    tree_filt = BallTree(pts_filt)
    logger.debug(f"🌲 Built BallTree for {pts_filt.shape[0]} filtered points")

    # Count neighbors within radius
    count_total = tree_all.query_radius(pts, r=radius, count_only=True)
    logger.debug("⏱ Completed total neighbor counts")
    count_filtered = tree_filt.query_radius(pts, r=radius, count_only=True)
    logger.debug("⏱ Completed filtered neighbor counts")

    # Compute fraction safely
    fraction = np.zeros_like(count_total, dtype=float)
    nonzero = count_total > 0
    fraction[nonzero] = count_filtered[nonzero] / count_total[nonzero]
    adata.obs["QC_fraction_filtered"] = fraction
    logger.debug(f"📊 QC_fraction_filtered: min={fraction.min():.4f}, max={fraction.max():.4f}")

    # Flagging using threshold
    thresh = 0.25
    flag = fraction > thresh
    adata.obs["QC_tsne_based_filter"] = flag
    logger.info(f"🎯 t-SNE filtering flagged {flag.sum()} cells at threshold={thresh}")

    # Update filtering_status
    to_update = flag & (adata.obs["filtering_status"] == "Unfiltered")
    adata.obs.loc[to_update, "filtering_status"] = "bad tsne cluster"
    logger.debug(f"🔄 Updated status for {to_update.sum()} cells")

    return adata
    x_vals, y_vals = coords[:, 0], coords[:, 1]

    filtered_mask = adata.obs["filtering_status"] != "Unfiltered"
    num_prefilter = int(filtered_mask.sum())
    logger.debug(f"🔍 Prefiltered cells (status != Unfiltered): {num_prefilter}")

    # Build KD-trees
    start_tree = time.time()
    tree_all = cKDTree(np.column_stack((x_vals, y_vals)))
    tree_filtered = cKDTree(np.column_stack((x_vals[filtered_mask], y_vals[filtered_mask])))
    dt_tree = time.time() - start_tree
    logger.debug(f"🌲 KD-tree construction took {dt_tree:.2f}s")

    # Query total neighborhoods
    pts = np.column_stack((x_vals, y_vals))
    start_all = time.time()
    neighbors_total = tree_all.query_ball_point(pts, r=radius)
    dt_all = time.time() - start_all
    logger.debug(f"⏱ Query total neighborhoods ({len(pts)} points) took {dt_all:.2f}s")

    # Query filtered neighborhoods
    start_filt = time.time()
    neighbors_filtered = tree_filtered.query_ball_point(pts, r=radius)
    dt_filt = time.time() - start_filt
    logger.debug(f"⏱ Query filtered neighborhoods ({len(pts)} points) took {dt_filt:.2f}s")

    # Compute counts and fraction
    count_total = np.fromiter((len(n) for n in neighbors_total), dtype=int)
    count_filtered = np.fromiter((len(n) for n in neighbors_filtered), dtype=int)
    logger.debug(f"🔢 Sample counts (first 5): total={count_total[:5]}, filtered={count_filtered[:5]}")

    fraction = np.zeros_like(count_total, dtype=float)
    valid = count_total > 0
    fraction[valid] = count_filtered[valid] / count_total[valid]
    adata.obs["QC_fraction_filtered"] = fraction
    logger.debug(f"📊 QC_fraction_filtered stats: min={fraction.min():.4f}, max={fraction.max():.4f}")

    tsne_thresh = 0.25
    mask_tsne = fraction > tsne_thresh
    adata.obs["QC_tsne_based_filter"] = mask_tsne
    num_flagged = int(mask_tsne.sum())
    logger.info(f"🎯 t-SNE filtering flagged {num_flagged} cells (threshold {tsne_thresh})")

    # Update filtering_status
    to_update = mask_tsne & (adata.obs["filtering_status"] == "Unfiltered")
    adata.obs.loc[to_update, "filtering_status"] = "bad tsne cluster"
    logger.debug(f"🔄 Updated filtering_status for {to_update.sum()} newly flagged cells.")

    return adata
    x_vals, y_vals = x[:, 0], x[:, 1]

    filtered_mask = adata.obs["filtering_status"] != "Unfiltered"
    num_filtered = filtered_mask.sum()
    logger.debug(f"🔍 t-SNE filtering: prefiltered cells={num_filtered}")

    # Build k-d trees
    tree_all = cKDTree(np.column_stack((x_vals, y_vals)))
    tree_filtered = cKDTree(np.column_stack((x_vals[filtered_mask], y_vals[filtered_mask])))
    logger.debug("🌲 KD-trees constructed")

    # Query neighborhoods
    pts = np.column_stack((x_vals, y_vals))
    neighbors_total = tree_all.query_ball_point(pts, r=radius)
    neighbors_filtered = tree_filtered.query_ball_point(pts, r=radius)
    logger.debug(
        f"📡 Queried {len(neighbors_total)} neighborhoods for total and {len(neighbors_filtered)} for filtered.")

    count_total = np.array([len(n) for n in neighbors_total], dtype=int)
    count_filtered = np.array([len(n) for n in neighbors_filtered], dtype=int)
    logger.debug(f"🔢 Example counts (total, filtered): {list(zip(count_total[:5], count_filtered[:5]))}")

    # Compute fraction
    fraction = np.zeros_like(count_total, dtype=float)
    valid = count_total > 0
    fraction[valid] = count_filtered[valid] / count_total[valid]
    adata.obs["QC_fraction_filtered"] = fraction
    logger.debug(f"📊 QC_fraction_filtered stats: min={fraction.min()}, max={fraction.max()}")

    tsne_thresh = 0.25
    mask_tsne = fraction > tsne_thresh
    adata.obs["QC_tsne_based_filter"] = mask_tsne
    num_flagged = mask_tsne.sum()
    logger.info(f"🎯 t-SNE filtering flagged {num_flagged} cells (threshold {tsne_thresh})")

    # Update filtering_status
    to_update = mask_tsne & (adata.obs["filtering_status"] == "Unfiltered")
    adata.obs.loc[to_update, "filtering_status"] = "bad tsne cluster"
    logger.debug(f"🔄 Updated filtering_status for {to_update.sum()} newly flagged cells.")

    return adata


def generate_segmentation_images(adata, fovs, density_dir, segmentation_dir, region_cells_by_fov):
    """
    Serial rendering of segmentation overlays with progress bar.
    """
    os.makedirs(segmentation_dir, exist_ok=True)
    for fov in tqdm(fovs, desc="🎨 Segmentation QC", unit="FOV"):
        spatial = adata.uns.get("spatial", {}).get(fov, {})
        if "segmentation" not in spatial:
            logger.warning(f"⚠️ Skipping {fov} – no segmentation mask.")
            continue

        seg_mask = spatial["segmentation"]
        H, W = seg_mask.shape

        low = set(adata.obs.query("fov == @fov and QC_low_intensity_filter").index)
        tsne = set(adata.obs.query("fov == @fov and QC_tsne_based_filter").index)
        region = region_cells_by_fov.get(fov, set())

        img = np.zeros((H, W, 3), np.uint8)
        for y in range(H):
            for x in range(W):
                lab = seg_mask[y, x]
                if lab == 0:
                    continue
                key = f"{fov}_{lab}"
                if key in low:
                    img[y, x] = (255, 0, 0)
                elif key in region:
                    img[y, x] = (0, 255, 0)
                elif key in tsne:
                    img[y, x] = (255, 255, 0)
                else:
                    img[y, x] = (255, 255, 255)

        out = os.path.join(segmentation_dir, f"{fov}.png")
        plt.imsave(out, img)

    logger.info("🎨 Segmentation overlays completed")


def create_storyboard(image_paths, output_path, cols=10, max_storyboard_width=10000, ppi=70):
    if not image_paths:
        logger.warning("⚠️ No images for storyboard.")
        return
    imgs = [Image.open(p) for p in image_paths]
    w, h = min(i.width for i in imgs), min(i.height for i in imgs)
    imgs = [i.resize((w, h), Image.LANCZOS) for i in imgs]

    rows = (len(imgs) + cols - 1) // cols
    out = Image.new("RGB", (min(cols * w, max_storyboard_width), rows * h), (255, 255, 255))
    for i, img in enumerate(imgs):
        x, y = (i % cols) * w, (i // cols) * h
        out.paste(img, (x, y))
    out.save(output_path, dpi=(ppi, ppi))


def generate_storyboards(qc_output_dir, density_dir, segmentation_dir, storyboard_dir, fovs):
    os.makedirs(storyboard_dir, exist_ok=True)
    segs = [os.path.join(segmentation_dir, f"{f}.png") for f in fovs]
    dens = [os.path.join(density_dir, f"{f}.png") for f in fovs]
    if segs:
        create_storyboard(segs, os.path.join(storyboard_dir, "segmentation_storyboard.png"))
    if dens:
        create_storyboard(dens, os.path.join(storyboard_dir, "density_storyboard.png"))


def generate_tsne_plot(adata, qc_output_dir):
    if "X_tsne" not in adata.obsm:
        logger.warning("⚠️ No t-SNE coords, skipping.")
        return

    x, y = adata.obsm["X_tsne"][:, 0], adata.obsm["X_tsne"][:, 1]
    status = adata.obs["filtering_status"]
    colors = np.full(len(x), "lightgray", object)
    colors[status == "low intensity cell"] = "red"
    colors[status == "low quality region"] = "green"
    colors[status == "bad tsne cluster"] = "yellow"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, c=colors, s=0.5, alpha=0.8)
    ax.set_title("t-SNE QC")
    ax.set_xlabel("t-SNE 1");
    ax.set_ylabel("t-SNE 2")
    ax.legend(handles=[
        plt.Line2D([], [], marker='o', color='w', markerfacecolor=m, label=l, markersize=6)
        for l, m in
        [("Low Intensity", "red"), ("Low Quality", "green"), ("t-SNE Cluster", "yellow"), ("Unfiltered", "lightgray")]
    ], title="Status", loc="upper right")

    os.makedirs(qc_output_dir, exist_ok=True)
    plt.savefig(os.path.join(qc_output_dir, "tsne_qc.png"), dpi=300, bbox_inches="tight")
    plt.close()


def generate_summary_tables(adata, qc_output_dir):
    total = len(adata.obs)
    status = adata.obs["filtering_status"]
    counts = {
        "Total Cells": total,
        "Low Intensity": (status == "low intensity cell").sum(),
        "Low Quality Region": (status == "low quality region").sum(),
        "t-SNE Filter": (status == "bad tsne cluster").sum()
    }
    summary = pd.DataFrame([
        {"Filter Step": k, "Absolute Count": v, "Percentage": v / total * 100}
        for k, v in counts.items()
    ])
    fov_stats = []
    for fov in adata.obs["fov"].unique():
        m = adata.obs["fov"] == fov
        fov_stats.append({
            "FOV": fov,
            **{k: (m & (status == k.lower().replace(" ", "_"))).sum() for k in counts}
        })
    pd.DataFrame(fov_stats).to_csv(os.path.join(qc_output_dir, "per_fov_stats.csv"), index=False)
    summary.to_csv(os.path.join(qc_output_dir, "overall_stats.csv"), index=False)


def run_qc_from_memory(args, adata):
    qc_out, dens_dir, seg_dir, sb_dir = create_directories(args.output_base_path)
    logger.info("🚀 Running QC filtering pipeline ...")
    # Echo QC hyperparameters
    logger.info(f"🔧 Low-intensity threshold: {args.low_intensity_threshold}")
    logger.info(f"🔧 Density filter window size: {args.qc_window_size}, stride: {args.qc_stride}")
    logger.info(f"🔧 Density threshold: {args.qc_density_threshold}, region threshold: {args.qc_region_threshold}")
    logger.info(f"🔧 t-SNE filtering radius: {args.radius_DRfilter}")

    # Step 1: Low-intensity filter
    tot = adata.layers["sum_unhuddle"].sum(axis=1)
    adata.obs["total_intensity"] = np.asarray(tot).flatten()
    adata.obs["QC_low_intensity_filter"] = adata.obs["total_intensity"] < args.low_intensity_threshold

    # Step 2: Density filtering (spatial QC)
    logger.info("🔍 Density QC starting")
    adata, region_map = perform_density_filtering(
        adata,
        qc_out,
        dens_dir,
        window_size=args.qc_window_size,
        stride=args.qc_stride,
        density_threshold=args.qc_density_threshold,
        region_threshold=args.qc_region_threshold,
        density_scale=tuple(args.qc_plot_density_scale),
    )
    logger.info("🔍 Density QC completed")

    # Label QC_filter_low_quality_region
    adata.obs.loc[adata.obs_names.isin(set().union(*region_map.values())), "QC_filter_low_quality_region"] = True

    # Initialize filtering_status
    adata.obs["filtering_status"] = "Unfiltered"
    adata.obs.loc[adata.obs["QC_filter_low_quality_region"], "filtering_status"] = "low quality region"
    adata.obs.loc[adata.obs["QC_low_intensity_filter"], "filtering_status"] = "low intensity cell"

    # Step 3: t-SNE / DR coords
    logger.info("🔄 Loading t-SNE / external DR coordinates ...")
    tsne_avail = False
    tsne_dir = os.path.join(args.output_base_path, "fitsne_coords")
    if "X_tsne" in adata.obsm:
        logger.info("✅ Preloaded t-SNE found")
        tsne_avail = True
    elif "X_external_dr" in adata.obsm:
        adata.obsm["X_tsne"] = adata.obsm["X_external_dr"]
        logger.info("🔄 External DR used as t-SNE coords")
        tsne_avail = True
    elif os.path.isdir(tsne_dir) and os.listdir(tsne_dir):
        adata = load_dimension_reduction_coords(tsne_dir, adata, args.coord_cols)
        logger.info("✅ Loaded DR coords from disk")
        tsne_avail = True
    else:
        logger.info("⚠️ No DR coords found; skipping t-SNE filtering")

    # Step 4: Optional t-SNE filtering
    if tsne_avail:
        logger.info("🎲 Running t-SNE based filtering")
        adata = perform_tsne_filtering(adata, args.radius_DRfilter)
        logger.info("🎲 t-SNE filtering completed")

    # Step 5: Segmentation overlays
    logger.info("🎨 Generating segmentation overlays in parallel...")
    generate_segmentation_images(adata, list(region_map.keys()), dens_dir, seg_dir, region_map)
    logger.info("🎨 Segmentation overlays completed")

    # Step 6: Create storyboards
    logger.info("📚 Creating storyboards")
    generate_storyboards(qc_out, dens_dir, seg_dir, sb_dir, list(region_map.keys()))
    logger.info("📚 Storyboards created")

    # Step 7: t-SNE plot
    if tsne_avail:
        logger.info("📈 Rendering t-SNE QC plot")
        generate_tsne_plot(adata, qc_out)
        logger.info("📈 t-SNE QC plot done")

    # Step 8: Summary tables
    logger.info("📊 Generating summary tables")
    generate_summary_tables(adata, qc_out)
    logger.info("📊 Summary tables written")

    # Final: Apply filtering and save
    logger.info("💾 Saving filtered AnnData")
    keep = ~(adata.obs["QC_low_intensity_filter"] | adata.obs["QC_filter_low_quality_region"])
    adata = adata[keep].copy()
    out_path = os.path.join(args.output_base_path, "adata_objects", "adata1.h5ad")
    adata.write_h5ad(out_path)
    logger.info(f"💾 QC-completed AnnData saved to: {out_path}")
    print(f"✅ QC-completed AnnData saved to: {out_path}")
    del adata
