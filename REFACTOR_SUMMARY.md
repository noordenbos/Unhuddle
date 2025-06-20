# UNHUDDLE Refactor: Complete Separation of Original and Denoised Branches

## Problem Statement

The original UNHUDDLE implementation had a critical design flaw where the `sum_unhuddle` layer data would change based on the `--use_denoised` flag. This violated the intended isolation between the canonical and experimental denoised analysis branches.

### Root Cause

The issue was in the `compute_reallocation_with_checks` function, which used a conditional parameter `use_denoised` to determine whether to use original or denoised intensities for calculating reallocation weights. This meant:

- **When `--use_denoised=False`**: `sum_unhuddle` reflected reallocation guided by original intensities
- **When `--use_denoised=True`**: `sum_unhuddle` reflected reallocation guided by denoised intensities

This created a hidden dependency that made the canonical branch inconsistent and unreliable.

## Solution: Complete Branch Separation

### 1. New Function Architecture

We've created three separate reallocation functions:

#### `compute_reallocation_original(interactions, protein_features, tol=1e-6)`
- **Purpose**: Canonical reallocation using ONLY original mean intensities
- **Guarantee**: `sum_unhuddle` data is always consistent regardless of denoising settings
- **Use Case**: Standard UNHUDDLE analysis branch

#### `compute_reallocation_denoised(interactions, protein_features, tol=1e-6)`
- **Purpose**: Experimental reallocation using ONLY denoised intensities
- **Guarantee**: Complete isolation from canonical branch
- **Use Case**: Experimental denoised analysis branch

#### `compute_reallocation_with_checks(interactions, protein_features, tol=1e-6, use_denoised=True)` (DEPRECATED)
- **Status**: Deprecated with warning
- **Reason**: Caused hidden interactions between branches

### 2. Updated Pipeline Flow

The new `process_fov_reallocation_only` function now:

1. **Always runs canonical reallocation** using `compute_reallocation_original()`
   - This ensures `sum_unhuddle` is always consistent
   - Uses original mean intensities for weight calculation
   - Outputs to `processed_data/unhuddle_sum/`

2. **Conditionally runs denoised reallocation** using `compute_reallocation_denoised()`
   - Only when `--use_denoised=True` and denoised data is available
   - Uses denoised intensities for weight calculation
   - Outputs to `processed_data/unhuddle_denoised_sum/`
   - Completely independent of canonical branch

### 3. Data Flow Guarantees

#### Canonical Branch (Always Active)
```
Original Intensities → compute_reallocation_original() → sum_unhuddle
```

#### Experimental Branch (Conditional)
```
Denoised Intensities → compute_reallocation_denoised() → sum_unhuddle_denoised
```

## Benefits

### 1. **Reproducibility**
- `sum_unhuddle` data is now guaranteed to be identical regardless of denoising settings
- Canonical branch provides consistent baseline for all analyses

### 2. **Isolation**
- Experimental denoised branch is completely independent
- No hidden interactions between branches
- Clear separation of concerns

### 3. **Backward Compatibility**
- Old function is deprecated but still functional
- Gradual migration path for existing code
- Clear deprecation warnings guide users

### 4. **Transparency**
- Explicit function names indicate their purpose
- Clear documentation of data flow
- Easy to understand and maintain

## Migration Guide

### For Users
- No action required for standard usage
- `sum_unhuddle` data will now be consistent across all runs
- Experimental denoised data available in `sum_unhuddle_denoised` layer when `--use_denoised=True`

### For Developers
- Replace calls to `compute_reallocation_with_checks()` with appropriate new function
- Use `compute_reallocation_original()` for canonical branch
- Use `compute_reallocation_denoised()` for experimental branch

## Testing Recommendations

1. **Verify Consistency**: Run the same dataset with and without `--use_denoised` and confirm `sum_unhuddle` is identical
2. **Validate Isolation**: Confirm that denoised branch data is independent of canonical branch
3. **Check Performance**: Ensure no performance degradation from the refactor
4. **Test Edge Cases**: Verify behavior with missing denoised data, empty datasets, etc.

## Future Considerations

1. **Function Naming**: Consider renaming functions to be more descriptive (e.g., `compute_reallocation_canonical`)
2. **Configuration**: Consider making the separation more explicit in configuration files
3. **Documentation**: Update user documentation to reflect the new architecture
4. **Validation**: Add automated tests to prevent regression of this issue 