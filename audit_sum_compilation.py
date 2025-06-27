#!/usr/bin/env python3
"""
Audit script to compare sum compilation methods between original and solo border approaches.
This will help identify why the values might be the same when they should be different.
"""

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def audit_sum_compilation_methods(base_path, fov_name):
    """
    Audit the sum compilation methods for a specific FOV.
    
    Parameters:
    -----------
    base_path : str
        Base path to the processed data
    fov_name : str
        Name of the FOV to audit
    """
    
    # Define paths
    dirs = {
        "protein": os.path.join(base_path, "processed_data", "protein"),
        "unhuddle_denoised_sum": os.path.join(base_path, "processed_data", "unhuddle_denoised_sum"),
        "unhuddle_denoised_sum_original_style": os.path.join(base_path, "processed_data", "unhuddle_denoised_sum_original_style"),
        "original_sum": os.path.join(base_path, "processed_data", "original_sum"),
    }
    
    logger.info(f"🔍 Auditing sum compilation methods for FOV: {fov_name}")
    
    # Load the data files
    protein_path = os.path.join(dirs["protein"], f"{fov_name}.csv")
    solo_border_path = os.path.join(dirs["unhuddle_denoised_sum"], f"{fov_name}.csv")
    original_style_path = os.path.join(dirs["unhuddle_denoised_sum_original_style"], f"{fov_name}.csv")
    original_sum_path = os.path.join(dirs["original_sum"], f"{fov_name}.csv")
    
    # Check if files exist
    files_to_check = {
        "protein": protein_path,
        "solo_border": solo_border_path,
        "original_style": original_style_path,
        "original_sum": original_sum_path
    }
    
    for name, path in files_to_check.items():
        if not os.path.exists(path):
            logger.error(f"❌ Missing file: {name} at {path}")
            return
        else:
            logger.info(f"✅ Found {name} file: {path}")
    
    # Load data
    protein_df = pd.read_csv(protein_path)
    solo_border_df = pd.read_csv(solo_border_path)
    original_style_df = pd.read_csv(original_style_path)
    original_sum_df = pd.read_csv(original_sum_path)
    
    logger.info(f"📊 Data shapes:")
    logger.info(f"  - Protein: {protein_df.shape}")
    logger.info(f"  - Solo border: {solo_border_df.shape}")
    logger.info(f"  - Original style: {original_style_df.shape}")
    logger.info(f"  - Original sum: {original_sum_df.shape}")
    
    # Get marker names (excluding Label column)
    markers = [col for col in solo_border_df.columns if col != "Label"]
    logger.info(f"🎯 Found {len(markers)} markers: {markers[:5]}...")
    
    # Compare values for each marker
    logger.info(f"\n🔍 Comparing values between methods:")
    
    for marker in markers[:10]:  # Check first 10 markers
        solo_vals = solo_border_df[marker].values
        orig_style_vals = original_style_df[marker].values
        
        # Basic statistics
        solo_mean = np.mean(solo_vals)
        orig_style_mean = np.mean(orig_style_vals)
        solo_std = np.std(solo_vals)
        orig_style_std = np.std(orig_style_vals)
        
        # Check if values are identical
        identical = np.allclose(solo_vals, orig_style_vals, rtol=1e-10)
        
        logger.info(f"\n  {marker}:")
        logger.info(f"    Solo border: mean={solo_mean:.4f}, std={solo_std:.4f}")
        logger.info(f"    Original style: mean={orig_style_mean:.4f}, std={orig_style_std:.4f}")
        logger.info(f"    Identical: {identical}")
        
        if not identical:
            diff = orig_style_vals - solo_vals
            logger.info(f"    Difference: mean={np.mean(diff):.4f}, std={np.std(diff):.4f}")
            logger.info(f"    Max difference: {np.max(np.abs(diff)):.4f}")
        
        # Check if original style is higher (as expected)
        if orig_style_mean > solo_mean:
            logger.info(f"    ✅ Original style is higher (as expected)")
        elif orig_style_mean < solo_mean:
            logger.info(f"    ❌ Original style is lower (unexpected)")
        else:
            logger.info(f"    ⚠️ Values are the same (suspicious)")
    
    # Analyze the data sources
    logger.info(f"\n🔍 Analyzing data sources:")
    
    # Check what columns are available in protein_df
    denoised_cols = [col for col in protein_df.columns if col.endswith("_ExclusionMembrane_Sum_Intensity_denoised")]
    cell_sum_cols = [col for col in protein_df.columns if col.endswith("_Cell_Sum_Intensity") and not col.endswith("_denoised")]
    
    logger.info(f"  Denoised exclusion membrane columns: {len(denoised_cols)}")
    logger.info(f"  Original cell sum columns: {len(cell_sum_cols)}")
    
    if denoised_cols:
        logger.info(f"  Sample denoised columns: {denoised_cols[:3]}")
    if cell_sum_cols:
        logger.info(f"  Sample cell sum columns: {cell_sum_cols[:3]}")
    
    # Check if there are any denoised cell sum columns
    denoised_cell_sum_cols = [col for col in protein_df.columns if col.endswith("_Cell_Sum_Intensity_denoised")]
    logger.info(f"  Denoised cell sum columns: {len(denoised_cell_sum_cols)}")
    if denoised_cell_sum_cols:
        logger.info(f"  Sample denoised cell sum columns: {denoised_cell_sum_cols[:3]}")
    
    # Compare a few specific cells
    logger.info(f"\n🔍 Detailed cell comparison (first 5 cells):")
    
    for i in range(min(5, len(solo_border_df))):
        cell_label = solo_border_df.iloc[i]["Label"]
        logger.info(f"\n  Cell {cell_label}:")
        
        for marker in markers[:3]:  # First 3 markers
            solo_val = solo_border_df.iloc[i][marker]
            orig_style_val = original_style_df.iloc[i][marker]
            
            # Get original cell sum value
            cell_sum_col = f"{marker}_Cell_Sum_Intensity"
            if cell_sum_col in protein_df.columns:
                cell_idx = protein_df[protein_df["Label"] == cell_label].index
                if len(cell_idx) > 0:
                    original_cell_val = protein_df.iloc[cell_idx[0]][cell_sum_col]
                    logger.info(f"    {marker}: solo={solo_val:.4f}, orig_style={orig_style_val:.4f}, orig_cell={original_cell_val:.4f}")
                else:
                    logger.info(f"    {marker}: solo={solo_val:.4f}, orig_style={orig_style_val:.4f}, orig_cell=NOT_FOUND")
            else:
                logger.info(f"    {marker}: solo={solo_val:.4f}, orig_style={orig_style_val:.4f}, orig_cell=NO_COLUMN")

def main():
    """Main function to run the audit."""
    
    # You can modify these paths as needed
    base_path = "."  # Current directory, modify as needed
    fov_name = "Tonsil1_1"  # Modify to match your FOV name
    
    logger.info("🚀 Starting sum compilation audit")
    logger.info(f"Base path: {base_path}")
    logger.info(f"FOV name: {fov_name}")
    
    try:
        audit_sum_compilation_methods(base_path, fov_name)
        logger.info("✅ Audit completed successfully")
    except Exception as e:
        logger.error(f"❌ Audit failed: {e}", exc_info=True)

if __name__ == "__main__":
    main() 