# src/unhuddle/cli.py
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import logging
from functools import partial
from unhuddle_denoise.cli_helpers import (
    parse_arguments,
    setup_logging,
    list_available_markers,
    setup_output_directories,
    get_fov_folders,
    build_feature_args,
    build_reallocation_args,
    run_parallel_stage,
    result_failed,
    create_adata,
    save_cli_call,
    fitsne,
    run_qc_pipeline,
    count_total_cells_from_csvs,
    run_cohort_normalization_adaptive,
    get_available_protein_markers,
)

# These must remain top-level for multiprocessing compatibility
def process_features(fov, dirs, args):
    from unhuddle_denoise.pipeline import process_fov_features_only
    return process_fov_features_only(*build_feature_args(fov, dirs, args))

def process_reallocation(fov, dirs, args):
    from unhuddle_denoise.pipeline import process_fov_reallocation_only
    return process_fov_reallocation_only(*build_reallocation_args(fov, dirs, args))

def main():
    args = parse_arguments()
    setup_logging(args.log_level, output_base_path=args.output_base_path)


    logger = logging.getLogger(__name__)
    logger.debug(f"Logger '{logger.name}' is active at level: {logging.getLevelName(logger.getEffectiveLevel())}")

    # Basic CLI validation
    if args.list_available_markers:
        list_available_markers(args)
        return

    if not args.normalisation_markers or not args.nuclear_markers:
        raise ValueError("Both --normalisation_markers and --nuclear_markers are required unless --list_available_markers is used.")

    if args.create_deepcell_mask and not args.geckodriver_path:
        raise ValueError("--geckodriver_path is required when --create_deepcell_mask is used.")

    if args.nuclear_markers_overlay is None:
        args.nuclear_markers_overlay = args.nuclear_markers
    logger.info(f"Using nuclear_markers_overlay: {args.nuclear_markers_overlay}")

    if args.membrane_markers_overlay is None:
        args.membrane_markers_overlay = args.normalisation_markers
    logger.info(f"Using membrane_markers_overlay: {args.membrane_markers_overlay}")
    if args.fitsne and args.add_dimensionreduction_coords:
        raise ValueError("Cannot use both --fitsne and --add_dimensionreduction_coords. Choose one.")

    if args.use_denoised:
        logger.info("⚙️ Percentile normalization enabled — cohort-level ExclMem_Sum data will be fetched before FOV loop.")

    # Setup paths and input FOVs
    dirs = setup_output_directories(args.output_base_path, args)

    fov_folders = get_fov_folders(args, dirs)

    if not fov_folders:
        print("❌ No FOVs selected for processing.")
        if not args.create_adata:
            return

        proceed = input("⚠️ No FOVs found. Do you still want to run only AnnData creation? (y/n): ").strip().lower()
        if proceed == "y":
            create_adata(args, dirs)
        else:
            print("🚫 Aborted AnnData creation.")
        return

    # Save CLI call for traceability
    cli_path = save_cli_call(args.output_base_path)
    logger.info(f"📝 CLI call saved to: {cli_path}")

    # Stage 1: Feature Extraction
    results_stage1 = run_parallel_stage(
        fov_folders,
        func=partial(process_features, dirs=dirs, args=args),
        max_workers=args.max_workers,
        description="🔬 Extracting Features"
    )
    # Count total number of cells after Stage 1
    args.total_cells = count_total_cells_from_csvs(dirs["protein"])
    logger.info(f"📊 Total number of cells in cohort: {args.total_cells:,}")

    if args.use_denoised and args.total_cells <= 100000:
        logger.warning(
            f"⚠️ Cohort contains only {args.total_cells:,} cells. "
            "Denoising may be unreliable below 100,000 cells."
        )
        if args.yes:
            logger.info("✅ Proceeding with denoising (auto-confirmed via --yes).")
        else:
            response = input("❓ Proceed with denoising anyway? (y/N): ").strip().lower()
            if response not in ["y", "yes"]:
                logger.info("⏭️ Skipping denoising due to small cohort size.")
                args.use_denoised = False
            else:
                logger.info("✅ Proceeding with denoising despite small cohort.")

    # Stage 2a: Denoising (cohort-level)
    if args.use_denoised:
        logger.info("📊 Computing denoised reallocation factors (cohort-wide) ...")
        from unhuddle_denoise.reallocation_denoiser import compute_denoised_reallocation_factors

        protein_csv_paths = [
            os.path.join(dirs["protein"], f)
            for f in os.listdir(dirs["protein"])
            if f.endswith(".csv")
        ]

        denoised_summary_path = os.path.join(dirs["QC_metadata_denoised"], "denoised_reallocation_summary.csv")
        compute_denoised_reallocation_factors(
            protein_csv_paths=protein_csv_paths,
            dirs=dirs
        )
        logger.info(f"✅ Denoised reallocation factors saved to: {denoised_summary_path}")

    # Stage 2b: Reallocation
    results_stage2 = run_parallel_stage(
        fov_folders,
        func=partial(process_reallocation, dirs=dirs, args=args),
        max_workers=args.max_workers,
        description="🔁 Reallocation"
    )

    successful = [os.path.basename(fov) for fov, res in results_stage2.items() if not result_failed(res)]
    if successful:
        print("📁 Processed FOV folders have updated masks and overlays — check the pseudocolored mask renders for validation.")
        print(f"📄 Unhuddle normalized output (partial): {dirs['unhuddle_norm']}")
        print(f"📄 Cell-level morphology metrics: {dirs['morph']}")
        print(f"📄 Raw/pre-normalization values: {args.output_base_path}\n")

    # Stage 2c: Cohort-level Normalization

    print("🔄 Running cohort-level normalization...")
    protein_features = get_available_protein_markers(
        base_path=args.base_path,
        nuclear_markers=args.nuclear_markers
    )

    logger.debug(f"🧬 Markers selected for normalization: {protein_features}")
    logger.debug(f"🧪 Sensor markers: {args.normalisation_markers}")

    run_cohort_normalization_adaptive(
        fov_folders=fov_folders,
        sum_dirs={
            'original': dirs['original_sum'],
            'unhuddle': dirs['unhuddle_sum'],
            'denoised': dirs.get('unhuddle_denoised_sum')
        },
        dirs=dirs,
        markers=protein_features,
        sensor_markers=args.normalisation_markers
    )

    # optional: run fitsne local (implementation complex, only advanced users)
    if args.fitsne:
        fitsne(args)

    # Optional: build AnnData
    if args.create_adata:
        create_adata(args, dirs)




if __name__ == "__main__":
    main()