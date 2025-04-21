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
    from scipy.spatial import cKDTree
    x, y = adata.obsm["X_tsne"][:, 0], adata.obsm["X_tsne"][:, 1]

    filtered = adata.obs["filtering_status"] != "Unfiltered"
    tree_all = cKDTree(np.column_stack((x, y)))
    tree_f = cKDTree(np.column_stack((x[filtered], y[filtered])))
    tot = [len(n) for n in tree_all.query_ball_point(np.column_stack((x, y)), r=radius)]
    flt = [len(n) for n in tree_f.query_ball_point(np.column_stack((x, y)), r=radius)]
    frac = np.divide(flt, tot, out=np.zeros_like(tot, float), where=np.array(tot) > 0)
    adata.obs["QC_fraction_filtered"] = frac
    adata.obs["QC_tsne_based_filter"] = frac > 0.25
    adata.obs.loc[adata.obs["QC_tsne_based_filter"] & (adata.obs["filtering_status"] == "Unfiltered"),
                  "filtering_status"] = "bad tsne cluster"
    return adata


def generate_segmentation_images(adata, fovs, density_dir, segmentation_dir, region_cells_by_fov):
    """
    Parallel rendering of segmentation overlays with progress bar.
    """
    os.makedirs(segmentation_dir, exist_ok=True)

    def render(fov):
        spatial = adata.uns.get("spatial", {}).get(fov, {})
        if "segmentation" not in spatial:
            logger.warning(f"⚠️ Skipping {fov} – no seg mask.")
            return

        seg_mask = spatial["segmentation"]
        H, W = seg_mask.shape

        low = set(adata.obs.query("fov == @fov and QC_low_intensity_filter").index)
        tsne = set(adata.obs.query("fov == @fov and QC_tsne_based_filter").index)
        region = region_cells_by_fov.get(fov, set())

        img = np.zeros((H, W, 3), np.uint8)
        for y in range(H):
            for x in range(W):
                lab = seg_mask[y, x]
                if not lab: continue
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
        return fov

    # parallel map with progress bar
    with ThreadPoolExecutor(max_workers=min(8, len(fovs))) as pool:
        list(tqdm(pool.map(render, fovs), total=len(fovs), desc="🎨 Segmentation QC"))


def create_storyboard(image_paths, output_path, cols=10, max_storyboard_width=10000, ppi=70):
    if not image_paths:
        logger.warning("⚠️ No images for storyboard.")
        return
    imgs = [Image.open(p) for p in image_paths]
    w, h = min(i.width for i in imgs), min(i.height for i in imgs)
    imgs = [i.resize((w, h), Image.LANCZOS) for i in imgs]

    rows = (len(imgs) + cols - 1) // cols
    out = Image.new("RGB", (min(cols*w, max_storyboard_width), rows*h), (255, 255, 255))
    for i, img in enumerate(imgs):
        x, y = (i % cols)*w, (i//cols)*h
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

    x, y = adata.obsm["X_tsne"][:,0], adata.obsm["X_tsne"][:,1]
    status = adata.obs["filtering_status"]
    colors = np.full(len(x), "lightgray", object)
    colors[status=="low intensity cell"] = "red"
    colors[status=="low quality region"] = "green"
    colors[status=="bad tsne cluster"] = "yellow"

    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(x, y, c=colors, s=0.5, alpha=0.8)
    ax.set_title("t-SNE QC")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(handles=[
        plt.Line2D([],[],marker='o', color='w', markerfacecolor=m, label=l, markersize=6)
        for l,m in [("Low Intensity","red"),("Low Quality","green"),("t-SNE Cluster","yellow"),("Unfiltered","lightgray")]
    ], title="Status", loc="upper right")

    os.makedirs(qc_output_dir, exist_ok=True)
    plt.savefig(os.path.join(qc_output_dir,"tsne_qc.png"), dpi=300, bbox_inches="tight")
    plt.close()


def generate_summary_tables(adata, qc_output_dir):
    total = len(adata.obs)
    status = adata.obs["filtering_status"]
    counts = {
        "Total Cells": total,
        "Low Intensity": (status=="low intensity cell").sum(),
        "Low Quality Region": (status=="low quality region").sum(),
        "t-SNE Filter": (status=="bad tsne cluster").sum()
    }
    summary = pd.DataFrame([
        {"Filter Step":k, "Absolute Count":v, "Percentage": v/total*100}
        for k,v in counts.items()
    ])
    fov_stats = []
    for fov in adata.obs["fov"].unique():
        m = adata.obs["fov"]==fov
        fov_stats.append({
            "FOV": fov,
            **{k:(m & (status==k.lower().replace(" ","_"))).sum() for k in counts}
        })
    pd.DataFrame(fov_stats).to_csv(os.path.join(qc_output_dir,"per_fov_stats.csv"), index=False)
    summary.to_csv(os.path.join(qc_output_dir,"overall_stats.csv"), index=False)


def run_qc_from_memory(args, adata):
    qc_out, dens_dir, seg_dir, sb_dir = create_directories(args.output_base_path)

    # Step 1
    tot = adata.layers["sum_unhuddle"].sum(axis=1)
    adata.obs["total_intensity"] = np.array(tot).flatten()
    adata.obs["QC_low_intensity_filter"] = adata.obs["total_intensity"] < args.low_intensity_threshold

    # Step 2: t-SNE coords (same as before)
    tsne_dir = os.path.join(args.output_base_path, "fitsne_coords")
    tsne_avail = False
    if "X_tsne" in adata.obsm:
        tsne_avail = True
    elif "X_external_dr" in adata.obsm:
        adata.obsm["X_tsne"] = adata.obsm["X_external_dr"]
        tsne_avail = True
    elif os.path.isdir(tsne_dir) and os.listdir(tsne_dir):
        adata = load_dimension_reduction_coords(tsne_dir, adata, args.coord_cols)
        tsne_avail = True

    adata.obs["filtering_status"] = "Unfiltered"

    # Step 4
    adata, region_map = perform_density_filtering(
        adata, qc_out, dens_dir,
        window_size=args.qc_window_size,
        stride=args.qc_stride,
        density_threshold=args.qc_density_threshold,
        region_threshold=args.qc_region_threshold,
        density_scale=tuple(args.qc_plot_density_scale)
    )
    adata.obs["QC_filter_low_quality_region"] = ~adata.obs_names.isin(
        set().union(*region_map.values())
    )

    # Step 5
    if tsne_avail:
        adata = perform_tsne_filtering(adata, args.radius_DRfilter)

    # Step 6 Labels
    adata.obs.loc[adata.obs["QC_filter_low_quality_region"], "filtering_status"] = "low quality region"
    adata.obs.loc[adata.obs["QC_low_intensity_filter"], "filtering_status"] = "low intensity cell"

    # Step 7
    fovs = list(region_map.keys())
    generate_segmentation_images(adata, fovs, dens_dir, seg_dir, region_map)
    generate_storyboards(qc_out, dens_dir, seg_dir, sb_dir, fovs)
    if tsne_avail:
        generate_tsne_plot(adata, qc_out)
    generate_summary_tables(adata, qc_out)

    # Step 8 & 9
    keep = ~(adata.obs["QC_low_intensity_filter"] | adata.obs["QC_filter_low_quality_region"])
    adata = adata[keep].copy()
    out_path = os.path.join(args.output_base_path, "adata_objects", "adata1.h5ad")
    adata.write_h5ad(out_path)
    print(f"✅ QC-completed AnnData saved to: {out_path}")
