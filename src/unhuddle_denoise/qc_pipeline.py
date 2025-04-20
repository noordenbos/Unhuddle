import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from PIL import Image
import logging
logger = logging.getLogger(__name__)


def create_directories(output_base_path):
    """
    Create standard output directories under a given base path.

    Returns:
        qc_output_dir: Path for QC plots
        density_dir: Path for density maps
        segmentation_dir: Path for storing segmentation previews
        storyboard_dir: Path for composite or annotated visuals
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


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from scipy.stats import gaussian_kde
from matplotlib.backends.backend_pdf import PdfPages


def compute_density_map(segmentation_mask, filtered_cells, window_size=50, stride=10):
    """
    Compute a density map over a segmentation mask.
    """
    height, width = segmentation_mask.shape
    density_map = np.zeros((height, width), dtype=np.float32)
    # Extract numeric portion from cell IDs in filtered_cells; assumes the cell name is "{fov}_{cellID}"
    filtered_cells_numeric = set(int(cell_id.split("_")[-1]) for cell_id in filtered_cells)
    if not any(cid in np.unique(segmentation_mask) for cid in filtered_cells_numeric):
        print("❌ No filtered cells found in segmentation mask! Possible index mismatch.")
        return density_map
    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            window = segmentation_mask[y:y + window_size, x:x + window_size]
            unique_ids = set(np.unique(window))
            unique_filtered = unique_ids.intersection(filtered_cells_numeric)
            count = len(unique_filtered)
            if count > 0:
                density_map[y:y + window_size, x:x + window_size] += count
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
    Loop over each FOV to compute density maps based on segmentation masks in adata.uns and
    update a flag for each cell if the fraction of pixels in a high-density region exceeds region_threshold.
    """
    filtered_cells_li = set(adata.obs.index[adata.obs["QC_low_intensity_filter"]])
    fovs = list(set(adata.obs["fov"]))
    region_cells_by_fov = {}
    adata.obs["QC_filter_low_quality_region"] = False

    for fov in fovs:
        logger.info(f"Processing density filter for FOV: {fov}")
        spatial_dict = adata.uns.get("spatial", {}).get(fov, {})
        if "segmentation" not in spatial_dict:
            logger.warning(f"⚠️ Skipping {fov} – No segmentation mask found.")
            continue
        segmentation_mask = spatial_dict["segmentation"]
        fov_cells = adata.obs.index[adata.obs["fov"] == fov]
        fov_filtered_cells = {cell for cell in filtered_cells_li if cell.startswith(f"{fov}_")}
        density_map = compute_density_map(segmentation_mask, fov_filtered_cells,
                                          window_size=window_size, stride=stride)

        plt.figure(figsize=(10, 10))
        plt.imshow(density_map, cmap='hot', interpolation='nearest', vmin=density_scale[0], vmax=density_scale[1])
        plt.axis('off')
        density_path = os.path.join(density_dir, f"{fov}.png")
        plt.savefig(density_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        logger.info(f"Saved density map for {fov} to: {density_path}")

        region_mask = density_map > density_threshold
        if region_mask.shape != segmentation_mask.shape:
            region_mask = resize(region_mask.astype(float), segmentation_mask.shape, anti_aliasing=True) > 0.5

        cell_pixel_counts = defaultdict(int)
        cell_region_counts = defaultdict(int)
        height, width = segmentation_mask.shape
        for y in range(height):
            for x in range(width):
                cell_id_num = segmentation_mask[y, x]
                if cell_id_num == 0:
                    continue
                cell_pixel_counts[cell_id_num] += 1
                if region_mask[y, x]:
                    cell_region_counts[cell_id_num] += 1

        region_cells = {f"{fov}_{cid}" for cid, count in cell_region_counts.items()
                        if (count / cell_pixel_counts[cid] > region_threshold)}
        region_cells_by_fov[fov] = region_cells
        adata.obs.loc[fov_cells, "QC_filter_low_quality_region"] = False
        adata.obs.loc[list(region_cells), "QC_filter_low_quality_region"] = True

    return adata, region_cells_by_fov

def perform_tsne_filtering(adata, radius):
    x = adata.obsm["X_tsne"][:, 0]
    y = adata.obsm["X_tsne"][:, 1]

    # Already flagged cells (not "Unfiltered")
    filtered_mask = adata.obs["filtering_status"] != "Unfiltered"
    x_filtered, y_filtered = x[filtered_mask], y[filtered_mask]

    tree_all = cKDTree(np.column_stack((x, y)))
    tree_filtered = cKDTree(np.column_stack((x_filtered, y_filtered)))
    neighbors_total = tree_all.query_ball_point(np.column_stack((x, y)), r=radius)
    count_total = np.array([len(neighbors) for neighbors in neighbors_total])
    neighbors_filtered = tree_filtered.query_ball_point(np.column_stack((x, y)), r=radius)
    count_filtered = np.array([len(neighbors) for neighbors in neighbors_filtered])
    fraction_filtered = np.zeros_like(count_total, dtype=float)
    valid = count_total > 0
    fraction_filtered[valid] = count_filtered[valid] / count_total[valid]
    adata.obs["QC_fraction_filtered"] = fraction_filtered

    tsne_threshold = 0.25
    adata.obs["QC_tsne_based_filter"] = adata.obs["QC_fraction_filtered"] > tsne_threshold
    print("✅ 'QC_fraction_filtered' and 'QC_tsne_based_filter' columns added to adata.obs.")
    # Update filtering_status for cells still unfiltered
    adata.obs.loc[
        (adata.obs["QC_tsne_based_filter"]) & (adata.obs["filtering_status"] == "Unfiltered"),
        "filtering_status"
    ] = "bad tsne cluster"
    return adata

def generate_segmentation_images(adata, fovs, density_dir, segmentation_dir, region_cells_by_fov):
    """
    Generate color-coded segmentation images for each FOV, labeling filtered cells by QC category.

    Colors:
        - Red: low intensity
        - Green: retained region cells (post density filter)
        - Yellow: t-SNE-based filter
        - White: unflagged cells

    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix with spatial masks in adata.uns['spatial'][fov]['segmentation']
    fovs : list[str]
        List of field-of-view IDs to process
    density_dir : str
        (Unused here; part of standard call signature)
    segmentation_dir : str
        Output directory for saving PNG masks
    region_cells_by_fov : dict
        Dict mapping FOVs to sets of retained cells after density filtering
    """
    os.makedirs(segmentation_dir, exist_ok=True)

    for fov in fovs:
        logger.info(f"🧩 Generating segmentation image for FOV: {fov}")
        spatial_dict = adata.uns.get("spatial", {}).get(fov, {})
        if "segmentation" not in spatial_dict:
            logger.warning(f"⚠️ Skipping {fov} – No segmentation mask found.")
            continue

        segmentation_mask = spatial_dict["segmentation"]
        height, width = segmentation_mask.shape

        # Per-FOV filtered cells
        fov_low_intensity = set(adata.obs.index[(adata.obs["fov"] == fov) & adata.obs.get("QC_low_intensity_filter", False)])
        fov_tsne_filtered = set(adata.obs.index[(adata.obs["fov"] == fov) & adata.obs.get("QC_tsne_based_filter", False)])
        fov_region = set(region_cells_by_fov.get(fov, set()))

        # Compose RGB image
        seg_img = np.zeros((height, width, 3), dtype=np.uint8)
        for yy in range(height):
            for xx in range(width):
                cell_id = segmentation_mask[yy, xx]
                if cell_id == 0:
                    continue
                cell_name = f"{fov}_{cell_id}"
                if cell_name in fov_low_intensity:
                    seg_img[yy, xx] = (255, 0, 0)       # Red
                elif cell_name in fov_region:
                    seg_img[yy, xx] = (0, 255, 0)       # Green
                elif cell_name in fov_tsne_filtered:
                    seg_img[yy, xx] = (255, 255, 0)     # Yellow
                else:
                    seg_img[yy, xx] = (255, 255, 255)   # White

        seg_path = os.path.join(segmentation_dir, f"{fov}.png")
        plt.imsave(seg_path, seg_img)
        logger.info(f"💾 Saved segmentation overlay to: {seg_path}")



def create_storyboard(image_paths, output_path, cols=10, max_storyboard_width=10000, ppi=70):
    if not image_paths:
        logger.warning("⚠️ No images provided to create_storyboard. Skipping.")
        return

    images = [Image.open(img) for img in image_paths]
    min_width = min(img.size[0] for img in images)
    min_height = min(img.size[1] for img in images)
    images = [img.resize((min_width, min_height), Image.LANCZOS) for img in images]

    rows = (len(images) + cols - 1) // cols
    storyboard_width = min(cols * min_width, max_storyboard_width)
    storyboard_height = rows * min_height
    storyboard = Image.new("RGB", (storyboard_width, storyboard_height), (255, 255, 255))

    for i, img in enumerate(images):
        x_offset = (i % cols) * min_width
        y_offset = (i // cols) * min_height
        storyboard.paste(img, (x_offset, y_offset))

    storyboard.save(output_path, dpi=(ppi, ppi))
    logger.info(f"🖼️ Storyboard saved to: {output_path}")
def generate_storyboards(qc_output_dir, density_dir, segmentation_dir, storyboard_dir, fovs):
    os.makedirs(storyboard_dir, exist_ok=True)

    segmentation_images = [
        os.path.join(segmentation_dir, f"{fov}.png")
        for fov in fovs if os.path.exists(os.path.join(segmentation_dir, f"{fov}.png"))
    ]
    density_images = [
        os.path.join(density_dir, f"{fov}.png")
        for fov in fovs if os.path.exists(os.path.join(density_dir, f"{fov}.png"))
    ]

    if segmentation_images:
        seg_path = os.path.join(storyboard_dir, "segmentation_storyboard.png")
        create_storyboard(segmentation_images, seg_path)
    else:
        logger.warning("⚠️ No segmentation images found to create storyboard.")

    if density_images:
        density_path = os.path.join(storyboard_dir, "density_storyboard.png")
        create_storyboard(density_images, density_path)
    else:
        logger.warning("⚠️ No density images found to create storyboard.")

def generate_tsne_plot(adata, qc_output_dir):
    if "X_tsne" not in adata.obsm:
        logger.warning("⚠️ t-SNE coordinates not found in adata.obsm['X_tsne']. Skipping plot.")
        return

    x = adata.obsm["X_tsne"][:, 0]
    y = adata.obsm["X_tsne"][:, 1]

    low_intensity_mask = adata.obs["filtering_status"] == "low intensity cell"
    low_quality_mask = adata.obs["filtering_status"] == "low quality region"
    tsne_filtered_mask = adata.obs["filtering_status"] == "bad tsne cluster"
    unfiltered_mask = ~(low_intensity_mask | low_quality_mask | tsne_filtered_mask)

    cell_colors = np.full(len(x), "lightgray", dtype=object)
    cell_colors[low_intensity_mask] = "red"
    cell_colors[low_quality_mask] = "green"
    cell_colors[tsne_filtered_mask] = "yellow"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, c=cell_colors, s=0.5, alpha=0.8)
    ax.set_title("t-SNE Visualization of Filtered Cells")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    for label, color in [
        ("Low Intensity Cell", "red"),
        ("Low Quality Region", "green"),
        ("t-SNE Filter > 0.25", "yellow"),
        ("Unfiltered", "lightgray"),
    ]:
        ax.scatter([], [], color=color, label=label, s=30)
    ax.legend(loc="upper right", title="Filtering Status")

    os.makedirs(qc_output_dir, exist_ok=True)
    png_path = os.path.join(qc_output_dir, "tsne_render_filters.png")
    pdf_path = os.path.join(qc_output_dir, "tsne_render_filters.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    logger.info(f"📈 t-SNE filter plot saved to: {png_path}")
def generate_summary_tables(adata, qc_output_dir):
    total_cells = len(adata.obs)

    low_intensity_mask = adata.obs["filtering_status"] == "low intensity cell"
    low_quality_mask = (adata.obs["filtering_status"] == "low quality region") & ~low_intensity_mask
    tsne_filtered_mask = (adata.obs["filtering_status"] == "bad tsne cluster") & ~low_intensity_mask & ~low_quality_mask

    low_intensity_count = low_intensity_mask.sum()
    low_quality_count = low_quality_mask.sum()
    tsne_filtered_count = tsne_filtered_mask.sum()
    cumulative_filtered = low_intensity_count + low_quality_count + tsne_filtered_count

    summary_df = pd.DataFrame({
        "Filter Step": ["Total Cells", "Low Intensity", "Low Quality Region", "t-SNE Based Filter", "Cumulative Filtered Total"],
        "Absolute Count": [total_cells, low_intensity_count, low_quality_count, tsne_filtered_count, cumulative_filtered]
    })
    summary_df["Percentage"] = (summary_df["Absolute Count"] / total_cells) * 100

    # Per-FOV breakdown
    fov_stats = []
    for fov in adata.obs["fov"].unique():
        fov_mask = adata.obs["fov"] == fov
        fov_total = fov_mask.sum()

        fov_low_intensity = (fov_mask & low_intensity_mask).sum()
        fov_low_quality = (fov_mask & low_quality_mask).sum()
        fov_tsne_filtered = (fov_mask & tsne_filtered_mask).sum()
        fov_cumulative = fov_low_intensity + fov_low_quality + fov_tsne_filtered

        pct_filtered = (fov_cumulative / fov_total) * 100 if fov_total > 0 else 0.0

        fov_stats.append({
            "FOV": fov,
            "Total Cells": fov_total,
            "Low Intensity": fov_low_intensity,
            "Low Quality Region": fov_low_quality,
            "t-SNE Based Filter": fov_tsne_filtered,
            "Cumulative Filtered Total": fov_cumulative,
            "Percentage Filtered (Cumulative)": pct_filtered
        })

    fov_df = pd.DataFrame(fov_stats)

    if "Percentage Filtered (Cumulative)" not in fov_df.columns:
        raise RuntimeError("❌ Missing 'Percentage Filtered (Cumulative)' column in per-FOV stats.")

    fov_df_sorted = fov_df.sort_values(by="Percentage Filtered (Cumulative)", ascending=False)

    os.makedirs(qc_output_dir, exist_ok=True)
    fov_df_sorted.to_csv(os.path.join(qc_output_dir, "per_fov_filtering_stats.csv"), index=False)
    summary_df.to_csv(os.path.join(qc_output_dir, "overall_filtering_stats.csv"), index=False)
    logger.info("📊 Filtering summary tables written to QC directory.")


def run_qc_from_memory(args, adata):
    output_base_path = args.output_base_path
    tsne_dir = os.path.join(output_base_path, "fitsne_coords")
    qc_output_dir, density_dir, segmentation_dir, storyboard_dir = create_directories(output_base_path)

    # Step 1: Compute total intensity and mark low-intensity cells
    total_intensity = adata.layers["sum_unhuddle"].sum(axis=1)
    adata.obs["total_intensity"] = np.asarray(total_intensity).flatten()
    adata.obs["QC_low_intensity_filter"] = adata.obs["total_intensity"] < args.low_intensity_threshold

    # Step 2: Optional t-SNE coords
    tsne_available = False
    if "X_tsne" in adata.obsm:
        logger.info("✅ Found preloaded t-SNE coordinates in adata.obsm['X_tsne']")
        tsne_available = True
    elif "X_external_dr" in adata.obsm:
        adata.obsm["X_tsne"] = adata.obsm["X_external_dr"]
        logger.info("🔄 Using external DR coords as t-SNE input (copied to adata.obsm['X_tsne'])")
        tsne_available = True
    elif os.path.exists(tsne_dir) and os.listdir(tsne_dir):
        adata = load_dimension_reduction_coords(tsne_dir, adata, args.coord_cols)
        tsne_available = True
    else:
        logger.warning("⚠️ No dimension reduction coordinates found. Skipping t-SNE-based filtering.")

    # Step 3: Init status
    adata.obs["filtering_status"] = "Unfiltered"

    # Step 4: Density filtering (return full AnnData, flag cells for removal)
    adata, region_cells_by_fov = perform_density_filtering(adata, qc_output_dir, density_dir)

    # Mark "bad" cells as low-density if not in region_cells_by_fov
    all_good_cells = set().union(*region_cells_by_fov.values())
    adata.obs["QC_filter_low_quality_region"] = ~adata.obs_names.isin(all_good_cells)

    # Step 5: Optional t-SNE filtering
    if tsne_available:
        adata = perform_tsne_filtering(adata, args.radius)

    # Step 6: Label filtering status
    adata.obs.loc[adata.obs["QC_filter_low_quality_region"], "filtering_status"] = "low quality region"
    adata.obs.loc[adata.obs["QC_low_intensity_filter"], "filtering_status"] = "low intensity cell"

    # Step 7: Generate visuals and outputs before dropping
    fovs = list(set(adata.obs["fov"]))
    generate_segmentation_images(adata, fovs, density_dir, segmentation_dir, region_cells_by_fov)
    generate_storyboards(qc_output_dir, density_dir, segmentation_dir, storyboard_dir, fovs)
    if tsne_available:
        generate_tsne_plot(adata, qc_output_dir)
    generate_summary_tables(adata, qc_output_dir)

    # Step 8: Apply final filtering for downstream use
    keep_mask = ~adata.obs["QC_low_intensity_filter"] & ~adata.obs["QC_filter_low_quality_region"]
    adata = adata[keep_mask].copy()

    # Step 9: Save output
    adata_output_path = os.path.join(output_base_path, "adata_objects", "adata1.h5ad")
    adata.write_h5ad(adata_output_path)
    print(f"✅ Updated AnnData saved in place to: {adata_output_path}")

    del adata

