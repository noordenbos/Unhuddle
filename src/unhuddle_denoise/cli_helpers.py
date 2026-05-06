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
from typing import Optional, List, Dict
from pathlib import Path
import platform
import subprocess


_LOGGING_INITIALIZED = False

def get_git_info():
    """Fetches git commit, branch, and tag information."""
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode('utf-8')
        try:
            # Suppress stderr to avoid "fatal: no tag exactly matches" message
            tag = subprocess.check_output(['git', 'describe', '--tags', '--exact-match'], 
                                        stderr=subprocess.DEVNULL).strip().decode('utf-8')
        except subprocess.CalledProcessError:
            tag = "N/A (not on a tag)"
        try:
            dirty = subprocess.check_output(['git', 'status', '--porcelain']).strip().decode('utf-8')
            status = "dirty" if dirty else "clean"
        except subprocess.CalledProcessError:
            status = "unknown"
        
        # Get remote tracking information
        try:
            remote_info = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{u}'], 
                                                stderr=subprocess.DEVNULL).strip().decode('utf-8')
            remote_branch = remote_info
            
            # Extract remote name and get its URL
            remote_name = remote_info.split('/')[0]  # e.g., "origin" from "origin/main"
            try:
                remote_url = subprocess.check_output(['git', 'remote', 'get-url', remote_name], 
                                                   stderr=subprocess.DEVNULL).strip().decode('utf-8')
            except subprocess.CalledProcessError:
                remote_url = "N/A (cannot get remote URL)"
        except subprocess.CalledProcessError:
            remote_branch = "N/A (no upstream branch)"
            remote_url = "N/A (no upstream branch)"
        
        # Get local vs remote status
        try:
            local_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
            remote_commit = subprocess.check_output(['git', 'rev-parse', remote_info], 
                                                  stderr=subprocess.DEVNULL).strip().decode('utf-8')
            
            if local_commit == remote_commit:
                sync_status = "up-to-date"
            else:
                # Check if local is ahead/behind
                ahead = subprocess.check_output(['git', 'rev-list', '--count', f'{remote_info}..HEAD'], 
                                              stderr=subprocess.DEVNULL).strip().decode('utf-8')
                behind = subprocess.check_output(['git', 'rev-list', '--count', f'HEAD..{remote_info}'], 
                                               stderr=subprocess.DEVNULL).strip().decode('utf-8')
                
                if ahead != '0' and behind != '0':
                    sync_status = f"diverged (+{ahead}/-{behind})"
                elif ahead != '0':
                    sync_status = f"ahead by {ahead}"
                else:
                    sync_status = f"behind by {behind}"
        except subprocess.CalledProcessError:
            sync_status = "N/A (cannot determine)"
            
        return commit, branch, tag, status, remote_branch, remote_url, sync_status
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "N/A", "N/A", "N/A", "N/A (git not found or not a git repo)", "N/A", "N/A", "N/A"

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
            console_handler.stream.reconfigure(encoding='utf-8', errors='ignore') # type: ignore
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
        
        # Add git info
        commit, branch, tag, status, remote_branch, remote_url, sync_status = get_git_info()
        root.info("📦 Version Control:")
        root.info(f"    Commit: {commit}")
        root.info(f"    Branch: {branch}")
        root.info(f"    Tag: {tag}")
        root.info(f"    Status: {status}")
        root.info(f"    Remote: {remote_branch}")
        root.info(f"    Remote URL: {remote_url}")
        root.info(f"    Sync: {sync_status}")

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
        help="Markers to use for DeepCell overlay (green channel - membrane/cytoplasm). If not provided, defaults to --normalization_markers."
    )
    parser.add_argument("--blue_markers", nargs="+", default=[], help="Optional markers for blue channel")

    parser.add_argument("--fovs", nargs="*", default=None,
                        help="List of specific FOV folders to process")
    parser.add_argument("--mask_pattern", type=str, nargs="+", default=["*_0.tiff"],
                        help="Glob pattern for mask files (default '*_0.tiff')")

    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="WARNING")
    parser.add_argument("--check_output_exist", action="store_true", default=False,
                        help="Skip FOVs if output already exists in normalization folder")
    parser.add_argument("--normalization_markers", nargs="*", default=["all"],
                        help="Total sensor markers to normalize in '--normalize sensormarkers' (e.g. housekeepers) Use 'all' to include all available (nuclear_markers are automatically excluded) markers (default).")
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
    parser.add_argument("--use_denoised", dest="use_denoised", action="store_true", default=True,
                        help="Uses cohort level data to denoise reallocation factors (enabled by default)")
    parser.add_argument("--no_denoise", dest="use_denoised", action="store_false",
                        help="Disable denoising (not recommended)")
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
    parser.add_argument("--percentile", type=float, default=5,
                        help="lower percentile noise filtering during the cohort denoising step (default: 5)")
    #silent:
    parser.add_argument("--qc_region_threshold", type=float, default=0.8,
                        help=argparse.SUPPRESS)
    parser.add_argument("--qc_window_size", type=int, default=50,
                        help=argparse.SUPPRESS)
    parser.add_argument("--qc_stride", type=int, default=10,
                        help=argparse.SUPPRESS)
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--normalization",
        choices=["area", "sensormarker"],
        default="sensormarker",
        help="Normalization method to use ('area' or total protein of 'sensormarker'). Default: sensormarker"
    )
    parser.add_argument(
        "--denoise_regress",
        choices=["area", "perimeter"],
        default="perimeter",
        help="Regress based on ('area' or 'perimeter'). Default: perimeter"
    )
    parser.add_argument(
        "--denoise_method",
        choices=["percentile", "noisecone"],
        default="noisecone",
        help="Denoise on general lower percentile per size bin (percentile, expert usage only) or regression on large noisy cells (noisecone, recommended). Default: noisecone"
    )
    parser.add_argument(
        "--denoise_x_anchor_multiplier",
        type=float,
        default=-1.0,
        help="Multiplier for standard deviation to calculate x-anchor in noise cone denoising. anchor_x = mean + multiplier * std. Default: -1.0 (1 std below mean)"
    )
    # Hidden advanced denoising parameters
    parser.add_argument(
        "--denoise_sd_multiplier",
        type=float,
        default=3.0,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--denoise_signal_q_low",
        type=float,
        default=95.0,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--denoise_signal_q_high",
        type=float,
        default=99.99,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--denoise_min_cells_per_bin",
        type=int,
        default=10,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--denoise_min_area",
        type=float,
        default=15.0,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--denoise_noise_q",
        type=float,
        default=75.0,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--save_reallocation_debug",
        action="store_true",
        help="Save reallocation dictionaries and solo border pixel data to JSON files for debugging and validation"
    )
    parser.add_argument(
        "--add_original_compiled_sum",
        action="store_true", default=True,
        help="Use original-style compilation (original_sum + reallocated - taken) as the default denoised method. This creates the standard layer in the AnnData object."
    )
    parser.add_argument(
        "--no_original_compiled_sum",
        dest="add_original_compiled_sum",
        action="store_false",
        help="Disable original-style compilation (not recommended)"
    )
    parser.add_argument(
        "--add_strong_denoiser",
        action="store_true", default=False,
        help="Add strong denoised sum data using solo-border compilation (denoised_residuals + reallocated + solo_border_pixels). Expert usage only - creates additional layer in AnnData object."
    )

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
    logger = logging.getLogger(__name__)
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


def resolve_normalization_markers(normalization_markers, base_path, nuclear_markers):
    """
    Resolve the normalization markers list.

    Parameters:
    - normalization_markers (list[str]): List of specified markers or ['all'].
    - base_path (str): Base path to the FOV folders.
    - nuclear_markers (list[str]): List of nuclear markers to exclude.

    Returns:
    - list[str]: Final list of markers to use.
    """
    if normalization_markers is None:
        return []

    if "all" in normalization_markers:
        # Fetch all available markers if 'all' is specified
        return get_available_protein_markers(base_path, nuclear_markers)

    # Otherwise, return the provided list
    return normalization_markers


def get_available_protein_markers(
    base_path: str, nuclear_markers: list[str]
) -> list[str]:
    """
    Return list of protein markers by scanning .ome.tiff files in the first FOV,
    excluding nuclear markers.
    """
    fov_folders = [
        os.path.join(base_path, folder)
        for folder in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, folder))
    ]
    if not fov_folders:
        raise FileNotFoundError("No FOV folders found in base path.")

    first_fov = fov_folders[0]
    ome_files = glob.glob(os.path.join(first_fov, "*.ome.tiff"))
    if not ome_files:
        raise FileNotFoundError(f"No .ome.tiff files found in {first_fov}")

    all_markers = [os.path.basename(f).replace(".ome.tiff", "") for f in ome_files]
    protein_markers = sorted(m for m in all_markers if m not in nuclear_markers)

    return protein_markers

def setup_output_directories(output_base: str, args) -> dict:
    """
    Conditionally create and return the necessary output directories based on CLI flags.
    """
    dirs = {
        "morph": os.path.join(output_base, "features", "morphology_features"),
        "protein": os.path.join(output_base, "features", "protein_features"),
        "original_tables": os.path.join(output_base, "processed_data", "original_tables"),
        "original_sum": os.path.join(output_base, "processed_data", "original_tables", "original_sum"),
        "normalized_original": os.path.join(output_base, "processed_data", "original_tables", "original_normalized"),
        "unhuddle_sum": os.path.join(output_base, "processed_data", "unhuddle_sum"),
        "normalized_unhuddle": os.path.join(output_base, "processed_data", "unhuddle_normalized"),
        "QC": os.path.join(output_base, "QC"),
        "QC_normstats": os.path.join(output_base, "QC", "normalization","norm_stats"),
    }
    # Conditional folders
    if getattr(args, "create_adata", False):
        dirs["adata"] = os.path.join(output_base, "adata_objects")
        dirs["QC_plot"] = os.path.join(dirs["QC"], "normalization", "norm_plots")

    if getattr(args, "fitsne", False):
        dirs["dr"] = os.path.join(output_base, "dr_coords")

    if getattr(args, "use_denoised", False):
        # Default denoised directories (original-style compilation)
        dirs["unhuddle_denoised_sum"] = os.path.join(output_base, "processed_data", "unhuddle_denoised_sum")
        dirs["normalized_unhuddle_denoised"] = os.path.join(output_base, "processed_data", "unhuddle_denoised_normalized")
        dirs["QC_metadata_denoised"] = os.path.join(output_base, "QC", "denoiser")
        
        # Add directory for strong denoiser (solo border method) if requested
        if getattr(args, "add_strong_denoiser", False):
            dirs["unhuddle_denoised_sum_strong"] = os.path.join(output_base, "processed_data", "unhuddle_denoised_sum_strong")
            dirs["normalized_unhuddle_denoised_strong"] = os.path.join(output_base, "processed_data", "unhuddle_denoised_normalized_strong")

    if getattr(args, "save_reallocation_debug", False):
        dirs["QC_reallocation"] = os.path.join(dirs["QC"], "reallocation")

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
            if not glob.glob(os.path.join(dirs["normalized_unhuddle"], f"{os.path.basename(f)}*"))
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
        dirs,
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
        args.membrane_markers_overlay,
    )


def build_reallocation_args(fov: str, dirs: dict, args: argparse.Namespace):
    protein_path = os.path.join(dirs["protein"], f"{os.path.basename(fov)}.csv")
    protein_df = pd.read_csv(protein_path)
    if args.use_denoised and ("unhuddle_denoised_sum" not in dirs or "normalized_unhuddle_denoised" not in dirs):
        import logging
        logging.warning("⚠️ --use_denoised was passed, but denoised directories are missing from `dirs`")

    return (
        fov,
        protein_df,
        dirs,
        args.normalization_markers,
        args.use_denoised,
        args.log_level,
        args.save_reallocation_debug,
        getattr(args, "add_strong_denoiser", False),
    )


def maybe_build_adata(args, dirs):
    from unhuddle_denoise.adata_builder import build_adata_from_outputs
    try:
        adata = build_adata_from_outputs(
            dirs=dirs,
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


def create_adata(args: argparse.Namespace, dirs) -> None:
    logging.info("\n📦 Creating unified AnnData object...")
    adata = maybe_build_adata(args, dirs)

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
    run_qc_pipeline(args, adata, dirs)
    return adata





def fitsne(args):
    print("🚀 Running FIt-SNE dimensionality reduction step...")
    from unhuddle_denoise.run_fitsne import run_fitsne_dimension_reduction

    fitsne_dir = os.path.join(args.output_base_path, "dr_coords")
    input_dir = os.path.join(args.output_base_path, "normalized_unhuddle_denoised")

    run_fitsne_dimension_reduction(
        output_base=args.output_base_path,
        input_dir=input_dir,
        fitsne_dir=fitsne_dir,
        perplexity=30,  # optionally expose this as a CLI argument
        threads=args.max_workers  # reuse the same parallelism
    )

    print(f"✅ FIt-SNE coordinates saved to {fitsne_dir}")


def run_qc_pipeline(args, adata, dirs):
    def log_pre_qc_adata_summary(adata, name="Pre-QC"):
        import numpy as np
        logger = logging.getLogger(__name__)
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
    run_qc_from_memory(args, adata, dirs)  # Correct in-memory call

from pathlib import Path
from typing import List, Dict, Optional
import logging
import numpy as np

from unhuddle_denoise.normalization import (
    _per_cell_normalize,
    compute_adaptive_marker_stats_from_cohort,
    apply_cohort_scaling,
)


def run_cohort_normalization_adaptive(
    fov_folders: List[str],
    sum_dirs: Dict[str, str],
    dirs: Dict[str, str],
    markers: List[str],
    sensor_markers: Optional[List[str]] = None,
    normalization: str = "sensormarker",
    sample_max_cells: int = 100_000,
    min_range: float = 1e-5,
    cv_frac_thresh: float = 0.01,
) -> None:
    """
    Perform adaptive, cohort-aware normalization with a unified scaling strategy per marker.

    Parameters:
        fov_folders: List of FOV folder paths.
        sum_dirs: Dictionary mapping branch names to input directories.
        dirs: Dictionary of output directories.
        markers: List of marker names.
        sensor_markers: Optional list of sensor markers for normalization.
        normalization: Normalization method ('area' or 'sensormarker').
        sample_max_cells: Max cells to sample for cohort stats.
        min_range: Minimum range for adaptive scaling.
        cv_frac_thresh: CV fraction threshold for adaptive scaling.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"🔄 Normalization method selected: {normalization}")

    # Validate normalization method
    if normalization not in ["area", "sensormarker"]:
        raise ValueError(f"Unknown normalization method: {normalization}")

    # Define branches and directories
    branch_to_outdir = {
        "original": dirs.get("normalized_original"),
        "unhuddle": dirs.get("normalized_unhuddle"),
        "denoised": dirs.get("normalized_unhuddle_denoised"),
        "denoised_strong": dirs.get("normalized_unhuddle_denoised_strong"),
    }

    # Process each branch
    for branch, out_dir in branch_to_outdir.items():
        in_dir = sum_dirs.get(branch)
        if not in_dir or not Path(in_dir).is_dir():
            logger.info(f"⚠️ Skipping branch '{branch}': input folder missing or invalid: {in_dir}")
            continue
        if not out_dir:
            logger.warning(f"⚠️ Skipping branch '{branch}': output directory not set")
            continue

        in_path = Path(in_dir)
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        fov_files = sorted(in_path.glob("*.csv"))
        if not fov_files:
            logger.warning(f"⚠️ No CSVs found in {in_dir} for branch '{branch}'")
            continue
        logger.info(f"🔄 Branch '{branch}': {len(fov_files)} FOVs to normalize")

        norms: List[np.ndarray] = []
        labels: List[pd.Series] = []
        fov_ids: List[str] = []
        areas: List[np.ndarray] = []  # Only used for area normalization

        # Stage 1: Read & per-cell normalize
        for csv_file in fov_files:
            try:
                cols_to_read = ["Label", *markers]
                df_markers = pd.read_csv(csv_file, usecols=cols_to_read)
            except Exception as e:
                logger.error(f"Failed to read '{csv_file.name}': {e}")
                continue

            if "Label" not in df_markers:
                logger.warning(f"⚠️ Missing 'Label' column in '{csv_file.name}'")
                continue

            mat = df_markers[markers].values

            if normalization == "sensormarker":
                # existing sensor-marker normalization logic
                if sensor_markers is None:
                    sensor_markers = markers.copy()
                else:
                    missing = set(sensor_markers) - set(markers)
                    if missing:
                        logger.warning(
                            "⚠️ The following sensor_markers are not in your marker list and will be ignored: %s",
                            sorted(missing),
                        )
                        sensor_markers = [m for m in sensor_markers if m in markers]

                try:
                    norm_mat, _ = _per_cell_normalize(mat, markers, sensor_markers)
                except Exception as e:
                    logger.error(f"Per-cell normalization failed for '{csv_file.stem}': {e}")
                    continue

            elif normalization == "area":
                # Read Area from morphological CSV
                morph_csv_file = Path(dirs['morph']) / csv_file.name  # assuming same filename
                if not morph_csv_file.is_file():
                    logger.error(f"⚠️ Morphological file '{morph_csv_file}' not found for area normalization.")
                    continue

                try:
                    df_morph = pd.read_csv(morph_csv_file, usecols=["Label", "Area"])
                except Exception as e:
                    logger.error(f"Failed to read morphological file '{morph_csv_file.name}': {e}")
                    continue

                # Merge Area into marker dataframe based on Label
                df_merged = pd.merge(df_markers, df_morph, on="Label", how="left")

                if df_merged["Area"].isnull().any():
                    missing_area_labels = df_merged.loc[df_merged["Area"].isnull(), "Label"].unique()
                    logger.error(f"⚠️ Missing Area values for labels {missing_area_labels} in '{csv_file.name}'")
                    continue

                area_values = df_merged["Area"].values
                if np.any(area_values <= 0):
                    logger.error(f"⚠️ Non-positive area values found in '{csv_file.name}'")
                    continue

                norm_mat = df_merged[markers].values / area_values[:, np.newaxis]
                areas.append(area_values)

                # Update labels and mat to reflect merged dataframe
                df_markers = df_merged
                mat = df_markers[markers].values

            norms.append(norm_mat)
            labels.append(df_markers["Label"])
            fov_ids.append(csv_file.stem)

        if not norms:
            logger.warning(f"⚠️ No valid FOVs after per-cell normalization for branch '{branch}'")
            continue

        # Stage 2: Compute cohort stats
        logger.info(f"📊 Computing cohort stats for branch '{branch}'")
        logger.debug(f"min_range: {min_range}")
        marker_stats = compute_adaptive_marker_stats_from_cohort(
            norms,
            var_names=markers,
            sample_max_cells=sample_max_cells,
            min_range=min_range,
            cv_frac_thresh=cv_frac_thresh,
        )

        # Stage 3: Apply cohort scaling per FOV
        logger.info(f"🎚️ Applying cohort scaling for branch '{branch}'")
        for idx, (fov, norm_mat, lbl) in enumerate(zip(fov_ids, norms, labels)):
            try:
                scaled = apply_cohort_scaling(norm_mat, markers, marker_stats)
            except Exception as e:
                logger.error(f"Scaling failed for '{fov}': {e}")
                continue

            out_df = pd.DataFrame(scaled, columns=markers)
            out_df.insert(0, "Label", lbl.values)

            out_file = out_path / f"{fov}.csv"
            out_df.to_csv(out_file, index=False)
            logger.debug(f"✅ Wrote normalized '{out_file}'")

        # Stage 4: Save QC stats
        qc_root = Path(dirs.get("QC_normstats", out_path.parent / "QC_normstats"))
        qc_root.mkdir(parents=True, exist_ok=True)
        stats_file = qc_root / f"{branch}_cohort_marker_qc.csv"
        marker_stats.to_csv(stats_file, index=False)
        logger.info(f"📈 Saved QC stats to '{stats_file}'")


def denoise_pipeline(args, dirs):
    '''
    Run the appropriate denoising method based on the user-specified flag.
    '''
    logger = logging.getLogger(__name__)
    denoise_method = args.denoise_method.lower()

    if denoise_method == "percentile":
        from unhuddle_denoise.percentile_denoise import run_percentile_denoise
        logger.info("📊 Running Percentile-based Denoising (cohort-wide) ...")
        run_percentile_denoise(args=args, dirs=dirs)
        logger.info("✅ Percentile denoising complete. QC & outputs in %s", dirs['QC_metadata_denoised'])

    elif denoise_method == "noisecone":
        from unhuddle_denoise.reallocation_denoiser import compute_denoised_reallocation_factors
        logger.info("📊 Running Noise cone-based Denoising (cohort-wide) ...")
        protein_csv_paths = [
            os.path.join(dirs["protein"], f)
            for f in os.listdir(dirs["protein"])
            if f.endswith(".csv")
        ]

        denoised_summary_path = os.path.join(dirs["QC_metadata_denoised"], "denoised_reallocation_summary.csv")

        compute_denoised_reallocation_factors(
            protein_csv_paths=protein_csv_paths,
            dirs=dirs,
            args=args
        )
        logger.info(f"✅ Denoised reallocation factors saved to: {denoised_summary_path}")

    else:
        logger.error(
            f"❌ Denoising method '{denoise_method}' not recognized. Please use 'percentile' or 'noisecone'.")




