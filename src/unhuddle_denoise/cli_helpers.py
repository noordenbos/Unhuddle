import os
import sys
import re
import glob
import logging
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from typing import Optional
import platform

_LOGGING_INITIALIZED = False

def setup_logging(log_level: str, output_base_path: Optional[str] = None) -> None:
    """
    Set up logging:
    - Console: logs at user-specified level (e.g., INFO or DEBUG)
    - File: logs INFO+ by default, or DEBUG if log_level == DEBUG
    Prevents reinitialization across FOV loop calls.
    """
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return

    requested_level = getattr(logging, log_level.upper(), logging.INFO)
    console_level = requested_level
    file_level = logging.DEBUG if requested_level == logging.DEBUG else logging.INFO

    # Reset logging and start clean
    logging.basicConfig(level=logging.NOTSET, force=True)
    root = logging.getLogger()
    root.handlers = []

    # ── Console Handler ───────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(console_handler)

    # ── Patch console stream for Unicode safety ───────
    if hasattr(console_handler.stream, 'reconfigure'):
        try:
            console_handler.stream.reconfigure(encoding='utf-8', errors='ignore')
        except Exception:
            pass  # Safe fallback if reconfigure not supported

    # ── File Handler (if requested) ───────────────────
    if output_base_path:
        log_dir = os.path.join(output_base_path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"unhuddle_run_{datetime.now():%Y%m%d_%H%M%S}.log")

        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root.addHandler(file_handler)

        # Write system & CLI summary
        root.info("📝 CLI call:\n    " + " ".join(sys.argv))
        root.info("🧪 Environment:")
        root.info(f"    Platform: {platform.platform()}")
        root.info(f"    Python version: {platform.python_version()}")
        root.info(f"    Executable: {sys.executable}")
        root.info(f"    CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        root.info(f"    NumPy version: {sys.modules.get('numpy', 'not loaded')}")
        root.info(f"    Working directory: {os.getcwd()}")
        root.info(f"📁 Full pipeline log will be saved to: {log_path}")

    # ── Optional Library Tuning (DEBUG only) ──────────
    if requested_level == logging.DEBUG:
        noisy_libs = [
            "selenium", "urllib3", "httpcore",
            "selenium.webdriver.remote.remote_connection"
        ]
        for lib in noisy_libs:
            lib_logger = logging.getLogger(lib)
            lib_logger.setLevel(logging.DEBUG)
            lib_logger.propagate = True

        for lib in ["matplotlib", "matplotlib.font_manager", "PIL", "PIL.Image"]:
            logging.getLogger(lib).setLevel(logging.WARNING)

    # ── Confirmation ──────────────────────────────────
    _LOGGING_INITIALIZED = True
    logger = logging.getLogger(__name__)
    logger.debug("🛠️ Logging system initialized (console=%s, file=%s)", logging.getLevelName(console_level), logging.getLevelName(file_level))

def parse_arguments() -> argparse.Namespace:
    """
    Parse and return the command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="UNHUDDLE: Deconvolute and normalize highly multiplex proteomics tissue data."
    )

    parser.add_argument("--base_path", type=str, required=True, help="Base path containing FOV folders")
    parser.add_argument("--output_base_path", type=str, required=True, help="Base path for output")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of parallel workers (default: 1)")

    parser.add_argument("--create_nuclear_mask", action="store_true", default=False,
                        help="Create nuclear mask for morphology and N/C ratio")
    parser.add_argument("--create_deepcell_mask", action="store_true", default=False,
                        help="Run DeepCell web overlay + segmentation")

    parser.add_argument("--geckodriver_path", type=str, default="geckodriver", help="Path to geckodriver binary")
    parser.add_argument("--deepcell_url", type=str, default="http://www.deepcell.org", help="DeepCell website URL")

    parser.add_argument("--nuclear_markers", nargs="+", default=None,
                        help="Filter marker list for chromatin signal: eg DNA1 DNA2 HistoneH3")
    parser.add_argument(
        "--nuclear_markers_overlay",
        nargs="+",
        default=None,
        help="Markers to use for DeepCell overlay (red channel - nuclear). If not provided, defaults to --nuclear_markers."
    )
    parser.add_argument(
        "--membrane_markers_overlay",
        nargs="+",
        default=None,
        help="Markers to use for DeepCell overlay (green channel - membrane/cytoplasm). If not provided, defaults to --normalisation_markers."
    )
    parser.add_argument("--blue_markers", nargs="+", default=[], help="Optional markers for blue channel")

    parser.add_argument("--fovs", nargs="*", default=None,
                        help="List of specific FOV folders to process")
    parser.add_argument("--mask_pattern", type=str, nargs="+", default=["*_0.tiff"],
                        help="Glob pattern for mask files (default '*_0.tiff')")

    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="WARNING")
    parser.add_argument("--check_output_exist", action="store_true", default=False,
                        help="Skip FOVs if output already exists in normalization folder")
    parser.add_argument("--normalisation_markers", nargs="*", default=None,
                        help="Sensor markers to normalize functional markers (e.g. CD3 CD45 Vimentin)")
    parser.add_argument("--list_available_markers", action="store_true",
                        help="Print available marker names from first FOV")
    parser.add_argument("--create_adata", action="store_true",
                        help="Integrates data from all FOVs in a single AnnData object")
    parser.add_argument(
        "--deepcell_resolution",
        type=int,
        choices=[10, 20, 40, 60, 100],
        default=10,
        help="Objective magnification to select in DeepCell UI (e.g., 10, 20, 40)"
    )
    parser.add_argument("--use_denoised", dest="use_denoised", action="store_true",
                        help="Experimental, uses cohort level data to denoise reallocation factors")
    parser.add_argument("--use_denoise", dest="use_denoised", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--fitsne", action="store_true",
                        help="Run dimension reduction using fitSNE and receive QC filtering")
    parser.add_argument("--no_qc", action="store_true", help="Avoid QC filtering on finalized AnnData object.")
    parser.add_argument("--add_dimensionreduction_coords", type=str, default=None,
                        help="Path to folder with {fov}.csv files having 3 columns: label, dr_1, dr_2")
    parser.add_argument("--coord_cols", nargs=2, type=str, default=["dr_1", "dr_2"],
                        help="Names of the coordinate columns in the CSVs (default: dr_1 dr_2)")
    parser.add_argument("--low_intensity_threshold", type=int, default=10,
                        help="Minimum total intensity to retain a cell (default: 10)")
    parser.add_argument("--qc_density_threshold", type=int, default=550,
                        help="Threshold for density map filtering (default: 550)")
    parser.add_argument("--qc_plot_density_scale", type=int, nargs=2, default=[0, 800],
                        help="Value range for density map visualization (default: 0 800)")
    parser.add_argument("--radius_DRfilter", type=float, default=0.8,
                        help="Neighborhood radius for dimension reduction based filtering (default: 0.8)")

    #silent:
    parser.add_argument("--qc_region_threshold", type=float, default=0.8,
                        help=argparse.SUPPRESS)
    parser.add_argument("--qc_window_size", type=int, default=50,
                        help=argparse.SUPPRESS)
    parser.add_argument("--qc_stride", type=int, default=10,
                        help=argparse.SUPPRESS)

    return parser.parse_args()

def save_cli_call(output_base_path, filename="cli_call.txt"):
    """Save the full CLI call to a file inside the output directory."""
    cmd = " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_output_path = os.path.join(output_base_path, filename)
    os.makedirs(output_base_path, exist_ok=True)
    with open(full_output_path, "w") as f:
        f.write(f"# Command run at {timestamp}\n")
        f.write(cmd + "\n")
    return full_output_path

def count_total_cells_from_csvs(csv_dir: str) -> int:
    """
    Count the total number of cells in a folder containing per-FOV CSVs
    (each row corresponds to a cell; assumes header row is present).
    """
    total = 0
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    for path in csv_files:
        try:
            with open(path, "r") as f:
                n_rows = sum(1 for _ in f) - 1  # subtract header
                total += n_rows
        except Exception as e:
            logger.warning(f"⚠️ Failed to count rows in {path}: {e}")
    return total

def list_available_markers(args: argparse.Namespace) -> None:
    """
    List available markers from the first FOV found in the base path.
    """
    base_path = args.base_path
    fov_folders = [
        os.path.join(base_path, folder)
        for folder in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, folder))
    ]
    if not fov_folders:
        print("❌ No FOV folders found.")
        return

    first_fov = fov_folders[0]
    ome_files = glob.glob(os.path.join(first_fov, "*.ome.tiff"))

    if not ome_files:
        print("❌ No .ome.tiff files in first FOV.")
        return

    marker_names = sorted([os.path.basename(f).replace(".ome.tiff", "") for f in ome_files])
    print(f"\nAvailable markers in FOV '{os.path.basename(first_fov)}':")
    print("list:", " ".join(marker_names))
    print("\n✅ Rerun without --list_available_markers to start the pipeline\n")


def setup_output_directories(output_base: str, args) -> dict:
    """
    Conditionally create and return the necessary output directories based on CLI flags.

    Parameters:
    -----------
    output_base : str
        Base path to create output subdirectories.
    args : Namespace
        Parsed CLI arguments to decide which folders are required.

    Returns:
    --------
    dict
        Mapping of folder roles to their full output paths.
    """

    dirs = {
        "morph": os.path.join(output_base, "morphology_features"),
        "protein": os.path.join(output_base, "protein_features"),
        "original_sum": os.path.join(output_base, "original_sum"),
        "original_norm": os.path.join(output_base, "original_normalized"),
        "unhuddle_sum": os.path.join(output_base, "unhuddle_sum"),
        "unhuddle_norm": os.path.join(output_base, "unhuddle_normalized"),
        "QC": os.path.join(output_base, "QC"),
        "QC_norm": os.path.join(output_base, "QC", "normalisation"),
    }

    # Conditional folders
    if getattr(args, "create_adata", False):
        dirs["adata"] = os.path.join(output_base, "adata_objects")

    if getattr(args, "fitsne", False) or getattr(args, "add_dimensionreduction_coords", None):
        dirs["dr"] = os.path.join(output_base, "dr_coords")

    if getattr(args, "use_denoised", False):
        dirs["unhuddle_denoised_sum"] = os.path.join(output_base, "unhuddle_denoised_sum")
        dirs["unhuddle_denoised_norm"] = os.path.join(output_base, "unhuddle_denoised_normalized")
        dirs["metadata_denoised"] = os.path.join(output_base, "metadata_denoise")

    # Actually create the folders
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    return dirs



def get_fov_folders(args: argparse.Namespace, dirs: dict) -> list:
    """
    Retrieve and filter FOV folders based on user-specified arguments.
    """
    all_fovs = [
        os.path.join(args.base_path, folder)
        for folder in os.listdir(args.base_path)
        if os.path.isdir(os.path.join(args.base_path, folder))
    ]
    fov_folders = all_fovs
    if args.fovs:
        fov_folders = [f for f in all_fovs if os.path.basename(f) in args.fovs]
    if args.check_output_exist:
        fov_folders = [
            f for f in fov_folders
            if not glob.glob(os.path.join(dirs["unhuddle_norm"], f"{os.path.basename(f)}*"))
        ]
    return fov_folders


def result_failed(res: dict) -> bool:
    """
    Check if a result indicates failure.
    """
    return any(
        "error" in key.lower() or ("deepcell" in key.lower() and "error" in str(value).lower())
        for key, value in res.items()
    )


def summarize_results(results: dict, stage_description: str = "") -> None:
    """
    Summarize and print processing results.
    """
    errored = [os.path.basename(fov) for fov, res in results.items() if result_failed(res)]
    successful = [os.path.basename(fov) for fov, res in results.items() if not result_failed(res)]

    print("\n" + "=" * 40)
    if stage_description:
        print(f"Summary for {stage_description}:")
    else:
        print("Processing Summary:")

    if not successful and errored:
        print("❗ All FOVs failed. If overlay exists (basepath/{fov}/overlay.png) and looks good, "
              "please check the DeepCell server and geckodriver path.")
    elif errored:
        print("⚠️ Some FOVs failed:")
        for fov in errored:
            print(f"   ❌ {fov}")
        print("⚠️ Tip: inspect overlay (if exists) basepath/{fov}/overlay.png\n")
        if successful:
            print("✅ Successfully processed FOVs:")
            for fov in successful:
                print(f"   {fov}")
            print()
    else:
        print("✅ All FOVs processed successfully.")
    print("=" * 40 + "\n")

from functools import partial

def dispatch_stage(fov_folders, process_fn, arg_builder_fn, args, dirs, description, max_workers):
    process = partial(process_fn, dirs=dirs, args=args)
    results = run_parallel_stage(
        fov_folders,
        func=lambda fov: process(fov),
        max_workers=max_workers,
        description=description
    )
    return results

def run_parallel_stage(fov_folders: list, func, max_workers: int, description: str) -> dict:
    """
    Run the given processing function in parallel for each FOV using ProcessPoolExecutor.
    """
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, fov): fov for fov in fov_folders}
        for future in tqdm(as_completed(futures), total=len(futures), desc=description):
            fov = futures[future]
            try:
                result = future.result()
                result["fov"] = fov
                results[fov] = result
            except Exception as e:
                logging.error(f"❌ FOV {fov} crashed: {e}")
                results[fov] = {"fov": fov, "critical_error": str(e), "crashed": True}
    summarize_results(results, stage_description=description)
    return results


def build_feature_args(fov: str, dirs: dict, args: argparse.Namespace):
    """
    Build arguments for the feature extraction stage for a given FOV.
    """
    return (
        fov,
        dirs["morph"],
        dirs["protein"],
        args.create_nuclear_mask,
        args.create_deepcell_mask,
        args.geckodriver_path,
        args.deepcell_url,
        args.mask_pattern,
        args.nuclear_markers,
        args.blue_markers,
        args.log_level,
        args.deepcell_resolution,
        args.nuclear_markers_overlay,
        args.membrane_markers_overlay

    )


def build_reallocation_args(fov: str, dirs: dict, args: argparse.Namespace):
    protein_path = os.path.join(dirs["protein"], f"{os.path.basename(fov)}.csv")
    protein_df = pd.read_csv(protein_path)
    if args.use_denoised and ("unhuddle_denoised_sum" not in dirs or "unhuddle_denoised_norm" not in dirs):
        import logging
        logging.warning("⚠️ --use_denoised was passed, but denoised directories are missing from `dirs`")

    return (
        fov,
        protein_df,
        dirs["original_sum"],
        dirs["original_norm"],
        dirs["unhuddle_sum"],
        dirs["unhuddle_norm"],
        dirs.get("unhuddle_denoised_sum", None),
        dirs.get("unhuddle_denoised_norm", None),
        args.normalisation_markers,
        args.use_denoised,
        args.log_level
    )


def maybe_build_adata(args):
    from unhuddle_denoise.adata_builder import build_adata_from_outputs
    try:
        adata = build_adata_from_outputs(
            output_base_path=args.output_base_path,
            working_path=args.base_path,
            output_adata_name="adata1.h5ad",
            max_workers=1
        )
        if adata is None:
            logging.error("❌ AnnData creation returned None.")
            return None
        return adata
    except Exception as e:
        logging.exception("❌ Exception during AnnData creation.")
        return None



def infer_dr_method_from_colnames(coord_cols):
    """
    Try to infer DR method name from coordinate column names.
    """
    known_methods = ["umap", "tsne", "fitsne", "optsne", "phate", "pca"]
    for col in coord_cols:
        for method in known_methods:
            if method in col.lower():
                return method
    return "ext_dr"  # fallback


def create_adata(args: argparse.Namespace) -> None:
    logging.info("\n📦 Creating unified AnnData object...")
    adata = maybe_build_adata(args)

    if adata is None:
        print("❌ AnnData creation failed. Skipping QC.")
        return

    if args.no_qc:
        print("⚠️ Skipping QC (user passed --no_qc)")
        return

    # ── Step 1: Defensive loading of fitsne ─────────────────────────────────────────
    fitsne_path = os.path.join(args.output_base_path, "dr_coords", f"{adata.obs['fov'].unique()[0]}.csv")
    if os.path.exists(fitsne_path):
        if not getattr(args, "fitsne", False):
            print("⚠️ Found dr_coords/ folder, but --fitsne was not passed.")
            print("ℹ️ Please rerun with --fitsne to use this folder, or remove the folder to avoid this message.")
            return

        coords = pd.read_csv(fitsne_path).values
        adata.obsm["X_fitsne"] = coords
        logging.info("✅ Loaded fitSNE coords into adata.obsm['X_fitsne']")

    # ── Step 2: Load external DR coordinates if supplied ────────────────────────────
    if args.add_dimensionreduction_coords:
        logging.info(f"📥 Loading external DR coordinates from: {args.add_dimensionreduction_coords}")
        all_coords = []

        for fov in adata.obs["fov"].unique():
            fov_csv = os.path.join(args.add_dimensionreduction_coords, f"{fov}.csv")
            if not os.path.exists(fov_csv):
                logging.warning(f"Skipping missing DR file: {fov_csv}")
                continue

            df = pd.read_csv(fov_csv)
            if df.shape[1] < 3:
                raise ValueError(f"Expected 3 columns in {fov_csv}, got: {df.columns.tolist()}")

            df.columns = ["label"] + args.coord_cols
            df["cell_id"] = df["label"].astype(str).apply(lambda x: f"{fov}_{x}")
            df = df.set_index("cell_id")

            valid_cells = adata.obs.index.intersection(df.index)
            missing = set(df.index) - set(valid_cells)
            if missing:
                logging.warning(f"{fov}: {len(missing)} unmatched DR coordinates")

            all_coords.append(df.loc[valid_cells, args.coord_cols])

        combined_coords = pd.concat(all_coords)
        coords_array = combined_coords.reindex(adata.obs.index).to_numpy()

        dr_method = infer_dr_method_from_colnames(args.coord_cols)
        obsm_key = f"X_{dr_method}"
        adata.obsm[obsm_key] = coords_array
        adata.uns["dr_source"] = f"external::{dr_method}"

        logging.info(f"✅ Stored external DR coordinates in adata.obsm['{obsm_key}']")
    run_qc_pipeline(args, adata)
    return adata





def fitsne(args):
    print("🚀 Running FIt-SNE dimensionality reduction step...")
    from unhuddle_denoise.run_fitsne import run_fitsne_dimension_reduction

    fitsne_dir = os.path.join(args.output_base_path, "dr_coords")
    input_dir = os.path.join(args.output_base_path, "unhuddle_denoised_normalized")

    run_fitsne_dimension_reduction(
        output_base=args.output_base_path,
        input_dir=input_dir,
        fitsne_dir=fitsne_dir,
        perplexity=30,  # optionally expose this as a CLI argument
        threads=args.max_workers  # reuse the same parallelism
    )

    print(f"✅ FIt-SNE coordinates saved to {fitsne_dir}")


def run_qc_pipeline(args, adata):
    def log_pre_qc_adata_summary(adata, name="Pre-QC"):
        import numpy as np
        logger = logging.getLogger("unhuddle")
        logger.info(f"🧬 Inspecting AnnData ({name})...")

        logger.info(f" - Total cells: {adata.n_obs}")
        logger.info(f" - Total markers: {adata.n_vars}")
        logger.info(f" - FOVs: {adata.obs['fov'].unique().tolist()}")

        if "sum_unhuddle" in adata.layers:
            intensity_sums = adata.layers["sum_unhuddle"].sum(axis=1).A1 if hasattr(adata.layers["sum_unhuddle"],
                                                                                    "A1") else adata.layers[
                "sum_unhuddle"].sum(axis=1)
            logger.info(
                f" - Intensity range across all cells: {np.min(intensity_sums):.4f} to {np.max(intensity_sums):.4f}")
            logger.info(f" - Cells with sum == 0: {(intensity_sums == 0).sum()}")

        if "X_spatial" in adata.obsm:
            coords = adata.obsm["X_spatial"]
            logger.info(
                f" - Spatial range: x=[{coords[:, 0].min():.1f}, {coords[:, 0].max():.1f}], y=[{coords[:, 1].min():.1f}, {coords[:, 1].max():.1f}]")
        else:
            logger.warning("⚠️ adata.obsm['X_spatial'] is missing.")

        missing_layers = [l for l in ["sum_unhuddle", "ExclMem_Sum"] if l not in adata.layers]
        if missing_layers:
            logger.warning(f"⚠️ Missing layers: {missing_layers}")

    log_pre_qc_adata_summary(adata, name="before QC")
    from unhuddle_denoise.qc_pipeline import run_qc_from_memory
    print("🚀 Running QC filtering pipeline ...")
    run_qc_from_memory(args, adata)  # Correct in-memory call

