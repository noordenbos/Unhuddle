# src/unhuddle/pipeline.py

import os
import logging
logger = logging.getLogger(__name__)
from skimage import io
from unhuddle_denoise.masks import process_cell_mask, process_nuclear_mask, process_membrane_masks, load_fov_files
from unhuddle_denoise.features import extract_morphology_features, extract_protein_intensity
from unhuddle_denoise.interactions import (
    compute_border_interactions,
    compute_background_interactions,
    merge_interactions,
    integrate_intensities_for_interactions,
    compute_reallocation_with_checks,
    compute_reallocation_original,
    compute_reallocation_denoised,
    settle_debts_intensity,
    settle_debts_from_residuals,
    settle_debts_from_residuals_original_style,
    save_reallocation_debug_data,
    compute_solo_border_pixels
)
from unhuddle_denoise.deepcell import create_deepcell_mask_overlay, process_deepcell_overlay

def process_fov_features_only(
    fov_path,
    dirs,
    create_nuclear_mask,
    create_deepcell_mask,
    geckodriver_path,
    deepcell_url,
    mask_pattern,
    nuclear_markers,
    blue_markers,
    log_level,
    deepcell_resolution,
    nuclear_markers_overlay,
    membrane_markers_overlay,
):
    from unhuddle_denoise.cli_helpers import setup_logging
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    logger.debug(f"[{os.getpid()}] Starting feature extraction for {fov_path}")
    result = {"fov": fov_path}

    try:
        if create_deepcell_mask:
            overlay_file = create_deepcell_mask_overlay(
                fov_path,
                red_markers=nuclear_markers_overlay,
                green_markers=membrane_markers_overlay,
                blue_markers=blue_markers
            )
            result["deepcell_overlay_file"] = overlay_file
            if overlay_file:
                try:
                    process_deepcell_overlay(
                        overlay_file,
                        fov_path,
                        deepcell_url,
                        geckodriver_path,
                        deepcell_resolution
                    )
                    result["deepcell_processing"] = "Success"
                except Exception as e:
                    logger.error(f"❌ DeepCell overlay processing failed: {e}")
                    result["deepcell_processing"] = f"Error: {e}"
                    return result
            else:
                logger.error("❌ Overlay file not created.")
                result["deepcell_processing"] = "Overlay file not created"
                return result

        files = load_fov_files(fov_path, nuclear_markers, mask_pattern)
        if "mask" not in files or not files["mask"]:
            logger.error(f"❌ No segmentation mask found in {fov_path} (pattern: {mask_pattern})")
            result["feature_extraction_error"] = "No mask found"
            return result

        cell_mask = process_cell_mask(fov_path, files["mask"])

        if create_nuclear_mask:
            try:
                nuclear_mask = process_nuclear_mask(fov_path, cell_mask, files, nuclear_markers)
                result["nuclear_mask_created"] = True
            except Exception as e:
                logger.warning(f"⚠️ Nuclear mask generation failed: {e}")
                nuclear_mask = None
                result["nuclear_mask_created"] = False
        else:
            nuclear_mask = None
            result["nuclear_mask_created"] = False

        try:
            membrane_mask, memexcl_mask = process_membrane_masks(fov_path, cell_mask)
            result["membrane_masks_generated"] = True
        except Exception as e:
            logger.error(f"❌ Membrane mask processing failed: {e}")
            result["feature_extraction_error"] = f"Membrane masks: {e}"
            return result

        try:
            morph_features = extract_morphology_features(fov_path, cell_mask, files, nuclear_markers, nuclear_mask, memexcl_mask, dirs=dirs)
            result["morphology_extracted"] = True
        except Exception as e:
            logger.error(f"❌ Morphology feature extraction failed: {e}")
            result["feature_extraction_error"] = f"Morphology: {e}"
            return result

        try:
            protein_features = extract_protein_intensity(
                fov_path, morph_features,
                cell_mask, membrane_mask, memexcl_mask, nuclear_markers, dirs=dirs
            )
            result["protein_intensity_extracted"] = True
        except Exception as e:
            logger.error(f"❌ Protein intensity extraction failed: {e}")
            result["feature_extraction_error"] = f"Protein intensity: {e}"
            return result

    except Exception as e:
        logger.error(f"❌ General feature extraction failed for {fov_path}: {e}")
        result["feature_extraction_error"] = str(e)

    return result

def process_fov_reallocation_only(
    fov_path,
    protein_features,
    dirs,
    markers_for_normalization,
    use_denoised,
    log_level,
    save_reallocation_debug=False,
    add_original_compiled_sum=False
):
    from unhuddle_denoise.cli_helpers import setup_logging
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    logger.debug(f"[{os.getpid()}] Starting reallocation for {fov_path}")
    result = {"fov": fov_path}

    try:
        # ── Load segmentation masks ─────────────────────────────────────
        cell_mask_path = os.path.join(fov_path, "deepcel_mask.tiff")
        if not os.path.exists(cell_mask_path):
            raise FileNotFoundError(f"deepcel_mask.tiff not found in {fov_path}")
        cell_mask = io.imread(cell_mask_path)

        membrane_path = os.path.join(fov_path, "membrane_mask.tiff")
        if not os.path.exists(membrane_path):
            raise FileNotFoundError(f"membrane_mask.tiff not found in {fov_path}")
        membrane_mask = io.imread(membrane_path)

        logger.debug(f"🧪 Loaded cell and membrane masks with shape: {cell_mask.shape}")

        # ── Compute interactions ────────────────────────────────────────
        border_int = compute_border_interactions(cell_mask, membrane_mask)
        background_int, _ = compute_background_interactions(cell_mask)
        merged = merge_interactions(border_int, background_int)
        all_interactions = integrate_intensities_for_interactions(fov_path, merged)

        # ── Canonical: Always run reallocation with original intensities ──
        logger.info(f"🔁 Running canonical reallocation (original intensities) for FOV: {fov_path}")
        reallocation_original = compute_reallocation_original(
            all_interactions, protein_features, tol=1e-6
        )
        original_sum_df, corrected_sum_df = settle_debts_intensity(
            fov_path, reallocation_original, protein_features, dirs
        )
        result["intensity_settled"] = True

        # ── Optional: Denoised Reallocation (separate branch) ─────────────
        reallocation_denoised = None
        solo_border_intensity = None
        
        if use_denoised and "unhuddle_denoised_sum" in dirs and "normalized_unhuddle_denoised" in dirs:
            logger.info(f"🔁 Running experimental denoised reallocation for FOV: {fov_path}")
            
            # Check if denoised data is available
            denoised_cols = [c for c in protein_features.columns if c.endswith("_ExclusionMembrane_Sum_Intensity_denoised")]
            if not denoised_cols:
                logger.warning(f"⚠️ No denoised columns found for {fov_path}, skipping denoised reallocation")
                result["denoised_skipped"] = "no_denoised_columns"
            else:
                # Use separate reallocation function with denoised intensities
                reallocation_denoised = compute_reallocation_denoised(
                    all_interactions, protein_features, tol=1e-6
                )
                
                # Always run solo-border method (standard)
                logger.info(f"🔁 Running standard solo-border method for denoised sum compilation")
                # Get solo border pixel data
                markers = [c.replace("_ExclusionMembrane_Sum_Intensity_denoised", "") for c in denoised_cols]
                solo_border_intensity = compute_solo_border_pixels(cell_mask, membrane_mask, fov_path, markers)
                
                denoised_df = settle_debts_from_residuals(
                    fov_folder=fov_path,
                    reallocation=reallocation_denoised,
                    protein_features=protein_features,
                    cell_mask=cell_mask,
                    membrane_mask=membrane_mask,
                    sensor_markers=markers_for_normalization,
                    dirs=dirs
                )
                
                # Optionally add original-style compilation as extra data
                if add_original_compiled_sum:
                    logger.info(f"🔁 Adding extra original-style denoised sum compilation")
                    original_style_df = settle_debts_from_residuals_original_style(
                        fov_folder=fov_path,
                        reallocation=reallocation_denoised,
                        protein_features=protein_features,
                        dirs=dirs
                    )
                    # Save to the separate directory for original-style compilation
                    fov_name = os.path.basename(fov_path)
                    original_style_path = os.path.join(dirs["unhuddle_denoised_sum_original_style"], f"{fov_name}.csv")
                    original_style_df.to_csv(original_style_path, index=False)
                    logger.info(f"📝 Saved extra original-style denoised sum to {original_style_path}")
                
                result["intensity_settled_denoised"] = True
        else:
            logger.info(f"ℹ️ Skipping denoised reallocation for {fov_path} — flag or output dirs not set.")

        # ── Save debug data if requested ──────────────────────────────────
        if save_reallocation_debug:
            fov_name = os.path.basename(fov_path)
            output_dir = dirs.get("QC_reallocation")
            if output_dir:
                logger.info(f"💾 Saving reallocation debug data for FOV: {fov_name}")
                save_reallocation_debug_data(
                    fov_name=fov_name,
                    reallocation_original=reallocation_original,
                    reallocation_denoised=reallocation_denoised,
                    solo_border_intensity=solo_border_intensity,
                    interactions=all_interactions,
                    output_dir=output_dir
                )
                result["debug_data_saved"] = True
            else:
                logger.warning(f"⚠️ QC_reallocation directory not found, skipping debug data save")

    except Exception as e:
        logger.error(f"❌ Reallocation failed for {fov_path}: {e}")
        result["reallocation_error"] = str(e)

    return result


