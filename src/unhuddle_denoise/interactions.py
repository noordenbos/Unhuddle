# src/unhudde/interactions.py

import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from skimage import io
import logging
import sys
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path
import platform
import json

logger = logging.getLogger(__name__)



def compute_border_interactions(cell_mask, membrane_mask):
    logger.debug(f"🧩 Starting border interaction computation")
    interactions = defaultdict(dict)
    membrane_indices = np.argwhere(membrane_mask > 0)[:, :2]
    logger.debug(f"🧩 Starting border interaction computation with {len(membrane_indices)} membrane pixels")

    for y, x in membrane_indices:
        current = cell_mask[y, x]
        if current == 0:
            continue  # Skip background pixels

        neighbors = set()
        for dy, dx in [(-1, 0), (0, -1), (0, 1), (1, 0),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < cell_mask.shape[0] and 0 <= nx < cell_mask.shape[1]:
                n = cell_mask[ny, nx]
                if n != 0 and n != current:
                    neighbors.add(n)

        if neighbors:
            interactions[f"{y}_{x}"] = {
                "current": int(current),
                "neighbors": sorted(map(int, neighbors)),
                "type": "border"
            }
    logger.debug(f"✅ Computed {len(interactions)} border interactions")
    return interactions


def compute_background_interactions(cell_mask):
    interactions = {}
    background_indices = np.argwhere(cell_mask == 0)[:, :2]
    neighbor_count = Counter()

    logger.debug(f"🌌 Scanning {len(background_indices)} background pixels for neighbors")

    for y, x in background_indices:
        labels = set()
        for dy, dx in [(-1, 0), (0, -1), (0, 1), (1, 0),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < cell_mask.shape[0] and 0 <= nx < cell_mask.shape[1]:
                n = cell_mask[ny, nx]
                if n != 0:
                    labels.add(n)

        if labels:
            interactions[f"{y}_{x}"] = {
                "current": -1,
                "interacts_with": sorted(map(int, labels)),
                "type": "background"
            }
            neighbor_count[len(labels)] += 1

    logger.debug(
        f"✅ Computed {len(interactions)} background interactions, with neighbor count breakdown: {dict(neighbor_count)}")
    return interactions, neighbor_count


def merge_interactions(border, background):
    merged = {**border, **background}
    logger.debug(f"🧬 Merged interactions: {len(border)} border + {len(background)} background = {len(merged)} total")
    return merged


def integrate_intensities_for_interactions(fov_folder, interactions):
    ome_tiffs = glob.glob(os.path.join(fov_folder, "*.ome.tiff"))
    logger.debug(f"🔬 Found {len(ome_tiffs)} OME-TIFFs in {fov_folder}")

    for ome_file in ome_tiffs:
        marker = os.path.basename(ome_file).replace(".ome.tiff", "")
        image = io.imread(ome_file)
        logger.debug(f"📷 Integrating intensities from {marker} with shape {image.shape}")

        for coord, data in interactions.items():
            try:
                y, x = map(int, coord.split("_"))
                if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                    value = float(image[y, x])
                else:
                    value = 0.0
            except Exception as e:
                logger.warning(f"⚠️ Failed to extract intensity for {marker} at {coord}: {e}")
                value = 0.0

            data.setdefault("intensities", {})[marker] = value

    logger.debug(f"✅ Integrated intensities for {len(interactions)} interactions")
    return interactions


def compute_reallocation_with_checks(interactions, protein_features, tol=1e-6, use_denoised=True):
    """
    DEPRECATED: This function has been replaced by compute_reallocation_original() and compute_reallocation_denoised().
    
    The old implementation caused hidden interactions between original and denoised branches.
    Please use the new separate functions for complete isolation.
    """
    import logging
    import numpy as np
    from collections import defaultdict
    import warnings

    warnings.warn(
        "compute_reallocation_with_checks is deprecated. "
        "Use compute_reallocation_original() for canonical branch and "
        "compute_reallocation_denoised() for experimental branch.",
        DeprecationWarning,
        stacklevel=2
    )

    logger = logging.getLogger(__name__)
    logger.debug("🧪 compute_reallocation_with_checks has been entered (DEPRECATED)")

    denoised_intensity = {}
    mean_intensity = {}

    for _, row in protein_features.iterrows():
        for col in row.index:
            if col.endswith("_ExclusionMembrane_Sum_Intensity_denoised"):
                marker = col.replace("_ExclusionMembrane_Sum_Intensity_denoised", "")
                denoised_intensity[(row["Label"], marker)] = row[col]
            elif col.endswith("_ExclusionMembrane_Mean_Intensity"):
                marker = col.replace("_ExclusionMembrane_Mean_Intensity", "")
                mean_intensity[(row["Label"], marker)] = row[col]

    def get_marker_intensity(label, marker):
        val = None
        if use_denoised:
            val = denoised_intensity.get((label, marker), None)
        if val is None:
            val = mean_intensity.get((label, marker), None)

        try:
            out = max(float(val), 0.0)
            if out == 0.0 and (val is None or (isinstance(val, float) and np.isnan(val))):
                logger.debug(f"⚠️ Intensity missing or NaN for {label}, {marker}")
            return out
        except Exception as e:
            logger.warning(f"⚠️ Invalid intensity for {label}, {marker}: {val} ({e})")
            return 0.0

    reallocation = defaultdict(lambda: {
        "taken_intensity": defaultdict(float),
        "reallocated_intensity": defaultdict(float)
    })

    for coord, data in interactions.items():
        if "intensities" not in data:
            continue

        interaction_type = data["type"]
        markers = [m for m in data["intensities"].keys() if "DNA" not in m and "Histone" not in m]

        if interaction_type == "border":
            current = data["current"]
            involved = [current] + data["neighbors"]
        else:
            current = None
            involved = data["interacts_with"]

        for marker in markers:
            values = [interactions[coord]["intensities"].get(marker, 0.0)]
            total = sum(values)

            if total == 0:
                continue  # skip unnecessary float ops

            weights = []
            valid_involved = []
            for i in involved:
                val = get_marker_intensity(i, marker)
                weights.append(val)
                valid_involved.append(i)

            try:
                denom = sum(weights)

                if denom > 0:
                    logger.debug(
                        f"🔢 Marker '{marker}' at {coord}: Weights = {weights}, Denom = {denom:.4f}, Total = {total:.2f}"
                    )
                    for i, w in zip(valid_involved, weights):
                        frac = w / denom
                        logger.debug(
                            f"↪️ Redistributing {frac:.4f} * {total:.2f} → Label={i}, marker={marker}"
                        )
                        reallocation[i]["reallocated_intensity"][marker] += frac * total
                        if interaction_type == "border" and i == current:
                            reallocation[i]["taken_intensity"][marker] += total
                else:
                    if interaction_type == "border" and isinstance(current, int):
                        reallocation[current]["reallocated_intensity"][marker] += total
                        reallocation[current]["taken_intensity"][marker] += total
            except Exception as e:
                logger.error(f"❌ Error during reallocation at {coord}, marker {marker}: {e}")

    return reallocation


def compute_reallocation_original(interactions, protein_features, tol=1e-6):
    """
    Compute reallocation using ONLY original mean intensities.
    This ensures sum_unhuddle is always consistent regardless of denoising settings.
    """
    import logging
    import numpy as np
    from collections import defaultdict

    logger = logging.getLogger(__name__)
    logger.debug("🧪 compute_reallocation_original: Using original intensities only")

    mean_intensity = {}

    for _, row in protein_features.iterrows():
        for col in row.index:
            if col.endswith("_ExclusionMembrane_Mean_Intensity"):
                marker = col.replace("_ExclusionMembrane_Mean_Intensity", "")
                mean_intensity[(row["Label"], marker)] = row[col]

    def get_marker_intensity(label, marker):
        val = mean_intensity.get((label, marker), None)
        try:
            out = max(float(val), 0.0)
            if out == 0.0 and (val is None or (isinstance(val, float) and np.isnan(val))):
                logger.debug(f"⚠️ Intensity missing or NaN for {label}, {marker}")
            return out
        except Exception as e:
            logger.warning(f"⚠️ Invalid intensity for {label}, {marker}: {val} ({e})")
            return 0.0

    reallocation = defaultdict(lambda: {
        "taken_intensity": defaultdict(float),
        "reallocated_intensity": defaultdict(float)
    })

    for coord, data in interactions.items():
        if "intensities" not in data:
            continue

        interaction_type = data["type"]
        markers = [m for m in data["intensities"].keys() if "DNA" not in m and "Histone" not in m]

        if interaction_type == "border":
            current = data["current"]
            involved = [current] + data["neighbors"]
        else:
            current = None
            involved = data["interacts_with"]

        for marker in markers:
            values = [interactions[coord]["intensities"].get(marker, 0.0)]
            total = sum(values)

            if total == 0:
                continue  # skip unnecessary float ops

            weights = []
            valid_involved = []
            for i in involved:
                val = get_marker_intensity(i, marker)
                weights.append(val)
                valid_involved.append(i)

            try:
                denom = sum(weights)

                if denom > 0:
                    logger.debug(
                        f"🔢 Marker '{marker}' at {coord}: Weights = {weights}, Denom = {denom:.4f}, Total = {total:.2f}"
                    )
                    for i, w in zip(valid_involved, weights):
                        frac = w / denom
                        logger.debug(
                            f"↪️ Redistributing {frac:.4f} * {total:.2f} → Label={i}, marker={marker}"
                        )
                        reallocation[i]["reallocated_intensity"][marker] += frac * total
                        if interaction_type == "border" and i == current:
                            reallocation[i]["taken_intensity"][marker] += total
                else:
                    if interaction_type == "border" and isinstance(current, int):
                        reallocation[current]["reallocated_intensity"][marker] += total
                        reallocation[current]["taken_intensity"][marker] += total
            except Exception as e:
                logger.error(f"❌ Error during reallocation at {coord}, marker {marker}: {e}")

    return reallocation


def compute_reallocation_denoised(interactions, protein_features, tol=1e-6):
    """
    Compute reallocation using ONLY denoised mean intensities.
    This is used for the experimental denoised branch only.
    """
    import logging
    import numpy as np
    from collections import defaultdict

    logger = logging.getLogger(__name__)
    logger.debug("🧪 compute_reallocation_denoised: Using denoised mean intensities only")

    denoised_intensity = {}

    for _, row in protein_features.iterrows():
        for col in row.index:
            if col.endswith("_ExclusionMembrane_Mean_Intensity_denoised"):
                marker = col.replace("_ExclusionMembrane_Mean_Intensity_denoised", "")
                denoised_intensity[(row["Label"], marker)] = row[col]

    def get_marker_intensity(label, marker):
        val = denoised_intensity.get((label, marker), None)
        try:
            out = max(float(val), 0.0)
            if out == 0.0 and (val is None or (isinstance(val, float) and np.isnan(val))):
                logger.debug(f"⚠️ Denoised mean intensity missing or NaN for {label}, {marker}")
            return out
        except Exception as e:
            logger.warning(f"⚠️ Invalid denoised mean intensity for {label}, {marker}: {val} ({e})")
            return 0.0

    reallocation = defaultdict(lambda: {
        "taken_intensity": defaultdict(float),
        "reallocated_intensity": defaultdict(float)
    })

    for coord, data in interactions.items():
        if "intensities" not in data:
            continue

        interaction_type = data["type"]
        markers = [m for m in data["intensities"].keys() if "DNA" not in m and "Histone" not in m]

        if interaction_type == "border":
            current = data["current"]
            involved = [current] + data["neighbors"]
        else:
            current = None
            involved = data["interacts_with"]

        for marker in markers:
            values = [interactions[coord]["intensities"].get(marker, 0.0)]
            total = sum(values)

            if total == 0:
                continue  # skip unnecessary float ops

            weights = []
            valid_involved = []
            for i in involved:
                val = get_marker_intensity(i, marker)
                weights.append(val)
                valid_involved.append(i)

            try:
                denom = sum(weights)

                if denom > 0:
                    logger.debug(
                        f"🔢 Marker '{marker}' at {coord}: Weights = {weights}, Denom = {denom:.4f}, Total = {total:.2f}"
                    )
                    for i, w in zip(valid_involved, weights):
                        frac = w / denom
                        logger.debug(
                            f"↪️ Redistributing {frac:.4f} * {total:.2f} → Label={i}, marker={marker}"
                        )
                        reallocation[i]["reallocated_intensity"][marker] += frac * total
                        if interaction_type == "border" and i == current:
                            reallocation[i]["taken_intensity"][marker] += total
                else:
                    if interaction_type == "border" and isinstance(current, int):
                        reallocation[current]["reallocated_intensity"][marker] += total
                        reallocation[current]["taken_intensity"][marker] += total
            except Exception as e:
                logger.error(f"❌ Error during reallocation at {coord}, marker {marker}: {e}")

    return reallocation


def compute_solo_border_pixels(cell_mask, membrane_mask, fov_folder, markers):
    logger = logging.getLogger(__name__)
    solo_pixels = defaultdict(list)
    border_coords = np.argwhere(membrane_mask > 0)

    for y, x in border_coords:
        current = cell_mask[y, x]
        if current == 0:
            continue

        neighbors = []
        for dy, dx in [(-1, 0), (0, -1), (0, 1), (1, 0),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < cell_mask.shape[0] and 0 <= nx < cell_mask.shape[1]:
                neighbors.append(cell_mask[ny, nx])

        # Accept if all neighbors are either background or same label
        if all(n in (0, current) for n in neighbors):
            solo_pixels[int(current)].append((y, x))

    logger.debug(f"🧭 Found {sum(len(v) for v in solo_pixels.values())} solo-border pixels across {len(solo_pixels)} cells")

    # Load all marker images from disk
    intensities_by_marker = {}
    for marker in markers:
        path = os.path.join(fov_folder, f"{marker}.ome.tiff")
        if os.path.exists(path):
            intensities_by_marker[marker] = io.imread(path).astype(np.float32)
        else:
            logger.warning(f"⚠️ Missing marker image for {marker} at {path}")

    # Compute intensity contributions
    result = defaultdict(lambda: defaultdict(float))
    for label, coords in solo_pixels.items():
        for y, x in coords:
            for marker, img in intensities_by_marker.items():
                result[label][marker] += float(img[y, x])

    return result


def settle_debts_intensity(fov_folder, reallocation, protein_features,
                           dirs):
    fov_name = os.path.basename(fov_folder)

    if not isinstance(protein_features.index, pd.MultiIndex):
        protein_features = protein_features.set_index(["FOV", "Label"])

    sum_cols = [c for c in protein_features.columns if c.endswith("_Cell_Sum_Intensity")]
    markers = [c.replace("_Cell_Sum_Intensity", "") for c in sum_cols]

    orig_df = protein_features.reset_index()[["Label"] + sum_cols]
    orig_df.columns = ["Label"] + markers
    orig_df.to_csv(os.path.join(dirs["original_sum"], f"{fov_name}.csv"), index=False)


    for label, d in reallocation.items():
        key = (fov_name, label)
        for m in markers:
            col = f"{m}_Cell_Sum_Intensity"
            if col in protein_features.columns:
                protein_features.at[key, col] -= d["taken_intensity"].get(m, 0)
                protein_features.at[key, col] += d["reallocated_intensity"].get(m, 0)

    corr_df = protein_features.reset_index()[["Label"] + sum_cols]
    corr_df.columns = ["Label"] + markers
    corr_df.to_csv(os.path.join(dirs["unhuddle_sum"], f"{fov_name}.csv"), index=False)


    return orig_df, corr_df

def settle_debts_from_residuals(
    fov_folder,
    reallocation,
    protein_features,
    cell_mask,
    membrane_mask,
    sensor_markers,
    dirs
):
    logger = logging.getLogger(__name__)
    fov_name = os.path.basename(fov_folder)
    if not isinstance(protein_features.index, pd.MultiIndex):
        protein_features = protein_features.set_index(["FOV", "Label"])
    residual_cols = [c for c in protein_features.columns if c.endswith("_ExclusionMembrane_Sum_Intensity_denoised")]
    markers = [c.replace("_ExclusionMembrane_Sum_Intensity_denoised", "") for c in residual_cols]

    for label, d in reallocation.items():
        key = (fov_name, label)
        for m in markers:
            col = f"{m}_ExclusionMembrane_Sum_Intensity_denoised"
            if col in protein_features.columns:
                protein_features.at[key, col] += d["reallocated_intensity"].get(m, 0)

    logger.debug("🧩 Adding solo-border pixel intensities to denoised residuals...")
    solo_border_intensity = compute_solo_border_pixels(cell_mask, membrane_mask, fov_folder, markers)

    for label, intensity_dict in solo_border_intensity.items():
        if not isinstance(intensity_dict, dict):
            logger.error(f"⚠️ Expected dict for intensity_dict, got {type(intensity_dict)} at label={label}")
            continue

        key = (fov_name, label)
        for m, value in intensity_dict.items():
            col = f"{m}_ExclusionMembrane_Sum_Intensity_denoised"
            if col in protein_features.columns:
                protein_features.at[key, col] += value
    corrected_df = protein_features.reset_index()[["Label"] + residual_cols]
    corrected_df.columns = ["Label"] + markers
    corrected_df.to_csv(os.path.join(dirs["unhuddle_denoised_sum"], f"{fov_name}.csv"), index=False)


    return corrected_df


def settle_debts_from_residuals_original_style(
    fov_folder,
    reallocation,
    protein_features,
    dirs
):
    """
    Alternative sum compilation method for denoised branch that follows the same pattern as canonical branch:
    sum_unhuddle_denoised = original_sum + reallocated_intensity - taken_intensity
    """
    logger = logging.getLogger(__name__)
    fov_name = os.path.basename(fov_folder)
    if not isinstance(protein_features.index, pd.MultiIndex):
        protein_features = protein_features.set_index(["FOV", "Label"])

    # Use original Cell_Sum_Intensity columns as base
    sum_cols = [c for c in protein_features.columns if c.endswith("_Cell_Sum_Intensity")]
    markers = [c.replace("_Cell_Sum_Intensity", "") for c in sum_cols]

    # Create a copy for the denoised sum calculation
    denoised_sum_df = protein_features.reset_index()[["Label"] + sum_cols].copy()
    denoised_sum_df.columns = ["Label"] + markers

    # Apply reallocation: original_sum + reallocated - taken
    for label, d in reallocation.items():
        key = (fov_name, label)
        for m in markers:
            col = f"{m}_Cell_Sum_Intensity"
            if col in protein_features.columns:
                original_val = protein_features.at[key, col]
                reallocated_val = d["reallocated_intensity"].get(m, 0)
                taken_val = d["taken_intensity"].get(m, 0)
                denoised_sum_df.loc[denoised_sum_df["Label"] == label, m] = original_val + reallocated_val - taken_val

    # Save to denoised sum directory
    denoised_sum_df.to_csv(os.path.join(dirs["unhuddle_denoised_sum"], f"{fov_name}.csv"), index=False)
    logger.info(f"📝 Saved original-style denoised sum to {fov_name}.csv")

    return denoised_sum_df


def save_reallocation_debug_data(
    fov_name: str,
    reallocation_original: dict,
    reallocation_denoised: Optional[dict] = None,
    solo_border_intensity: Optional[dict] = None,
    interactions: Optional[dict] = None,
    output_dir: str = None
) -> None:
    """
    Save reallocation dictionaries and solo border pixel data to JSON files for debugging and validation.
    
    Parameters:
    -----------
    fov_name : str
        Name of the FOV being processed
    reallocation_original : dict
        Reallocation dictionary from canonical branch
    reallocation_denoised : Optional[dict]
        Reallocation dictionary from experimental branch (if --use_denoised=True)
    solo_border_intensity : Optional[dict]
        Solo border pixel intensity data (if --use_denoised=True)
    interactions : Optional[dict]
        Raw interaction data for reference
    output_dir : str
        Directory to save the JSON files
    """
    if output_dir is None:
        logger.warning("⚠️ No output directory specified for reallocation debug data")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert defaultdict to regular dict for JSON serialization
    def convert_defaultdict(obj):
        if isinstance(obj, defaultdict):
            return dict(obj)
        elif isinstance(obj, dict):
            return {k: convert_defaultdict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_defaultdict(item) for item in obj]
        else:
            return obj
    
    # Prepare canonical reallocation data
    canonical_data = {
        "fov_name": fov_name,
        "timestamp": datetime.now().isoformat(),
        "reallocation_type": "canonical",
        "reallocation_data": convert_defaultdict(reallocation_original),
        "summary": {
            "total_cells": len(reallocation_original),
            "cells_with_taken_intensity": sum(1 for cell_data in reallocation_original.values() 
                                            if any(cell_data["taken_intensity"].values())),
            "cells_with_reallocated_intensity": sum(1 for cell_data in reallocation_original.values() 
                                                  if any(cell_data["reallocated_intensity"].values())),
            "total_taken_intensity": sum(sum(cell_data["taken_intensity"].values()) 
                                       for cell_data in reallocation_original.values()),
            "total_reallocated_intensity": sum(sum(cell_data["reallocated_intensity"].values()) 
                                             for cell_data in reallocation_original.values())
        }
    }
    
    # Save canonical reallocation data
    canonical_path = os.path.join(output_dir, f"{fov_name}_canonical_reallocation.json")
    with open(canonical_path, 'w') as f:
        json.dump(canonical_data, f, indent=2, default=str)
    logger.info(f"💾 Saved canonical reallocation data to: {canonical_path}")
    
    # Prepare and save denoised reallocation data if available
    if reallocation_denoised is not None:
        denoised_data = {
            "fov_name": fov_name,
            "timestamp": datetime.now().isoformat(),
            "reallocation_type": "denoised",
            "reallocation_data": convert_defaultdict(reallocation_denoised),
            "summary": {
                "total_cells": len(reallocation_denoised),
                "cells_with_taken_intensity": sum(1 for cell_data in reallocation_denoised.values() 
                                                if any(cell_data["taken_intensity"].values())),
                "cells_with_reallocated_intensity": sum(1 for cell_data in reallocation_denoised.values() 
                                                      if any(cell_data["reallocated_intensity"].values())),
                "total_taken_intensity": sum(sum(cell_data["taken_intensity"].values()) 
                                           for cell_data in reallocation_denoised.values()),
                "total_reallocated_intensity": sum(sum(cell_data["reallocated_intensity"].values()) 
                                                 for cell_data in reallocation_denoised.values())
            }
        }
        
        denoised_path = os.path.join(output_dir, f"{fov_name}_denoised_reallocation.json")
        with open(denoised_path, 'w') as f:
            json.dump(denoised_data, f, indent=2, default=str)
        logger.info(f"💾 Saved denoised reallocation data to: {denoised_path}")
    
    # Prepare and save solo border pixel data if available
    if solo_border_intensity is not None:
        solo_data = {
            "fov_name": fov_name,
            "timestamp": datetime.now().isoformat(),
            "data_type": "solo_border_pixels",
            "solo_border_data": convert_defaultdict(solo_border_intensity),
            "summary": {
                "total_cells_with_solo_pixels": len(solo_border_intensity),
                "total_solo_pixel_intensity": sum(sum(cell_data.values()) 
                                                for cell_data in solo_border_intensity.values())
            }
        }
        
        solo_path = os.path.join(output_dir, f"{fov_name}_solo_border_pixels.json")
        with open(solo_path, 'w') as f:
            json.dump(solo_data, f, indent=2, default=str)
        logger.info(f"💾 Saved solo border pixel data to: {solo_path}")
    
    # Prepare and save interaction data if available (for reference)
    if interactions is not None:
        # Sample a subset of interactions to avoid huge files
        sample_size = min(1000, len(interactions))
        sample_keys = list(interactions.keys())[:sample_size]
        sample_interactions = {k: interactions[k] for k in sample_keys}
        
        interaction_data = {
            "fov_name": fov_name,
            "timestamp": datetime.now().isoformat(),
            "data_type": "interactions_sample",
            "total_interactions": len(interactions),
            "sample_size": sample_size,
            "interaction_sample": convert_defaultdict(sample_interactions)
        }
        
        interaction_path = os.path.join(output_dir, f"{fov_name}_interactions_sample.json")
        with open(interaction_path, 'w') as f:
            json.dump(interaction_data, f, indent=2, default=str)
        logger.info(f"💾 Saved interaction sample data to: {interaction_path}")
    
    # Create a summary file
    summary_data = {
        "fov_name": fov_name,
        "timestamp": datetime.now().isoformat(),
        "files_generated": {
            "canonical_reallocation": os.path.basename(canonical_path),
            "denoised_reallocation": os.path.basename(denoised_path) if reallocation_denoised else None,
            "solo_border_pixels": os.path.basename(solo_path) if solo_border_intensity else None,
            "interactions_sample": os.path.basename(interaction_path) if interactions else None
        },
        "canonical_summary": canonical_data["summary"],
        "denoised_summary": denoised_data["summary"] if reallocation_denoised else None,
        "solo_border_summary": solo_data["summary"] if solo_border_intensity else None
    }
    
    summary_path = os.path.join(output_dir, f"{fov_name}_reallocation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    logger.info(f"💾 Saved reallocation summary to: {summary_path}")

