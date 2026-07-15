import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import label, find_objects
import pandas as pd



# ── Configure ───────────────────────────────────────────────────────────
CLEAN_PATH          = "clean_vol_Beef_Patties_no_STD.tiff"
SEEDED_PATH         = "clean_vol_Beef_Patties_w_STD.tiff"
BINS                = 1024
NOISE_SIGNAL_MAX    = 300
BASELINE_SIGMA      = 3.0   # std devs above left tail mean to set baseline
SAFETY_MARGIN       = 0.05  # pull threshold down by this fraction as safety buffer
SMOOTH_SIGNAL_WIDTH = 25    # smoothing width in signal intensity units
SPARSE_COUNT        = 50    # voxels per bin below which product is considered ended

raw_data_plot  = False
diff_data_plot = True
# ────────────────────────────────────────────────────────────────────────

RECIPE_PATH = r"C:\Users\kselvaganesan\Documents\CT_Recipe_Updates\Recipes_Unique_List.xlsx"
SHEET_NAME  = "Master_Unique_List"

def load_recipe_library(path, sheet_name=SHEET_NAME):
    df = pd.read_excel(path, sheet_name=sheet_name)
    numeric_cols = ["MainTh", "MinClusterDensityDiff", "MinClusterDensityDiffZ",
                     "MinClusterVolume"]
    for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.dropna(subset=["MainTh"]).reset_index(drop=True)
    print(f"  Loaded {len(df)} threshold-based recipes from {path}")
    return df

def rank_recipes_by_proximity(df, target_threshold):
    """
    Rank recipes by |MainTh - target_threshold|, ascending.
    Ties are broken by preferring '_high' variants over '_low' variants,
    per the requirement to always try high before low at a given threshold.
    """
    df = df.copy()
    df["distance"] = (df["MainTh"] - target_threshold).abs()
 
    # Preference score: high=0, low=1, neither=0.5 — sorts high first on ties
    def variant_rank(name):
        name = str(name).lower()
        if "high" in name:
            return 0
        elif "low" in name:
            return 1
        else:
            return 0.5
    df["variant_rank"] = df["Recipe_Name"].apply(variant_rank)
    df = df.sort_values(by=["distance", "variant_rank"]).reset_index(drop=True)
    return df

def calculate_comp_values(threshold, seeded_data):
    slag_mask = seeded_data >= threshold
    labeled_array, n_components = label(slag_mask)

    # Structuring element that dilates ONLY within the XY plane (no z spread)
    structure_xy = np.zeros((3, 3, 3), dtype=bool)
    structure_xy[1] = ndimage.generate_binary_structure(2, 1)  # in-plane cross

    Z = seeded_data.shape[0]
    component_properties = []

    for comp_id in range(1, n_components + 1):
        comp_mask    = labeled_array == comp_id
        voxel_coords = np.argwhere(comp_mask)
        voxel_values = seeded_data[comp_mask]

        volume       = int(comp_mask.sum())
        mean_density = float(voxel_values.mean())

        centroid_z = float(voxel_coords[:, 0].mean())
        centroid_y = float(voxel_coords[:, 1].mean())
        centroid_x = float(voxel_coords[:, 2].mean())

        # --- Shared per-slice interior stats (no loop) ---
        interior_sum   = (seeded_data * comp_mask).sum(axis=(1, 2))
        interior_count = comp_mask.sum(axis=(1, 2))
        interior_mean  = np.divide(interior_sum, interior_count,
                                    out=np.zeros(Z), where=interior_count > 0)
        interior_max   = np.where(
            comp_mask, seeded_data, -np.inf
        ).max(axis=(1, 2))

        valid_slice = interior_count > 0  # z-indices the component actually occupies

        # ---------------- Density Difference in XY ----------------
        dilated   = ndimage.binary_dilation(comp_mask, structure=structure_xy)
        ring_mask = dilated & ~comp_mask & ~slag_mask

        ring_sum   = (seeded_data * ring_mask).sum(axis=(1, 2))
        ring_count = ring_mask.sum(axis=(1, 2))
        ring_mean  = np.divide(ring_sum, ring_count,
                                out=np.full(Z, np.nan), where=ring_count > 0)

        xy_valid = valid_slice & (ring_count > 0)
        xy_mean_diffs = interior_mean[xy_valid] - ring_mean[xy_valid]
        xy_peak_diffs = interior_max[xy_valid]  - ring_mean[xy_valid]

        density_diff_xy_mean = float(np.mean(xy_mean_diffs)) if xy_mean_diffs.size else 0.0
        density_diff_xy_peak = float(np.mean(xy_peak_diffs)) if xy_peak_diffs.size else 0.0

        # ---------------- Density Difference in Z ----------------
        # Shift data/slag up and down by one slice (no python loop over z)
        below_data = np.zeros_like(seeded_data); below_data[1:] = seeded_data[:-1]
        above_data = np.zeros_like(seeded_data); above_data[:-1] = seeded_data[1:]
        below_slag = np.zeros_like(slag_mask);   below_slag[1:] = slag_mask[:-1]
        above_slag = np.zeros_like(slag_mask);   above_slag[:-1] = slag_mask[1:]

        below_exists = np.zeros(Z, dtype=bool); below_exists[1:]  = True
        above_exists = np.zeros(Z, dtype=bool); above_exists[:-1] = True

        valid_below = comp_mask & ~below_slag & below_exists[:, None, None]
        valid_above = comp_mask & ~above_slag & above_exists[:, None, None]

        z_bg_sum   = (below_data * valid_below).sum(axis=(1, 2)) + (above_data * valid_above).sum(axis=(1, 2))
        z_bg_count = valid_below.sum(axis=(1, 2)) + valid_above.sum(axis=(1, 2))
        z_bg_mean  = np.divide(z_bg_sum, z_bg_count,
                                out=np.full(Z, np.nan), where=z_bg_count > 0)

        z_valid = valid_slice & (z_bg_count > 0)
        z_mean_diffs = interior_mean[z_valid] - z_bg_mean[z_valid]
        z_peak_diffs = interior_max[z_valid]  - z_bg_mean[z_valid]

        density_diff_z_mean = float(np.mean(z_mean_diffs)) if z_mean_diffs.size else 0.0
        density_diff_z_peak = float(np.mean(z_peak_diffs)) if z_peak_diffs.size else 0.0


        if mean_density == 0.0 or density_diff_xy_mean == 0.0 or density_diff_z_mean == 0.0:
            continue

        component_properties.append({
            "id":              comp_id,
            "volume_voxels":   volume,
            "mean_density":    mean_density,
            "density_diff_z":  density_diff_z_mean,
            "density_diff_xy": density_diff_xy_mean,
            "centroid_z":      centroid_z,
            "centroid_y":      centroid_y,
            "centroid_x":      centroid_x,
        })

    return pd.DataFrame(component_properties), labeled_array, n_components

def component_matches_recipe(component_properties, recipe_row):
    """
    Check whether ANY component in comp_df satisfies a recipe's detection
    criteria. All five conditions must hold simultaneously for a single
    component — this mirrors the recipe's 'minimum' style thresholds
    used in the actual inspection software.

    Returns the subset of comp_df that passes all criteria (empty
    DataFrame if none pass).
    """
    matches = []

    for _, component_row in component_properties.iterrows():
        passes = (
            component_row["volume_voxels"]        >= recipe_row["MinClusterVolume"]        and
            component_row["density_diff_xy"]       >= recipe_row["MinClusterDensityDiff"]   and
            component_row["density_diff_z"]        >= recipe_row["MinClusterDensityDiffZ"]  and
            component_row["mean_density"]          >= recipe_row["MinClusterDensityTh"]    
        )
        if passes:
            matches.append(component_row)

    return pd.DataFrame(matches)

# ── Load ────────────────────────────────────────────────────────────────
print("Loading clean scan ...")
clean_data = tifffile.imread(CLEAN_PATH)

print("Loading seeded scan ...")
seeded_data = tifffile.imread(SEEDED_PATH)

# ── Crop to matching shape (handles off-by-one slice count differences) ───
if clean_data.shape != seeded_data.shape:
    min_slices  = min(clean_data.shape[0], seeded_data.shape[0])
    clean_data  = clean_data[:min_slices]
    seeded_data = seeded_data[:min_slices]
    print(f"  shapes differed — cropped both to: {clean_data.shape}")

assert clean_data.shape == seeded_data.shape, "Shapes still don't match after crop"

# ── Mask out negative values and plot histogram ─────────
clean_data_masked  = clean_data[clean_data >= 0]
seeded_data_masked = seeded_data[seeded_data >= 0]

bins = 256
shared_range = (min(clean_data_masked.min(), seeded_data_masked.min()),
                      max(clean_data_masked.max(), seeded_data_masked.max()))
counts, edges = np.histogram(clean_data_masked.ravel(), bins=bins, range=shared_range)
seeded_counts, _   = np.histogram(seeded_data_masked.ravel(), bins=bins, range=shared_range)
centers = 0.5 * (edges[:-1] + edges[1:])

# ── 1. Product peak ───────────────────────────────────────────────────────
## Uses a peak detection algorithm to find the location of the product peak
product_mask = centers > NOISE_SIGNAL_MAX
peaks, _         = find_peaks(counts[product_mask].astype(float),
                               prominence=counts[product_mask].max() * 0.05)
main_peak_idx    = np.where(product_mask)[0][peaks[np.argmax(counts[product_mask][peaks])]]
main_peak_signal = centers[main_peak_idx]

print(f"\n── Product peak ─────────────────────────────")
print(f"  peak signal:  {main_peak_signal:.0f}")
print(f"  peak count:   {counts[main_peak_idx]:,}")

# ── 2. Piecewise linear breakpoint detection for contaminant detection ────

## Prepare seeded tail in log space 
tail_mask     = centers > main_peak_signal
tail_centers  = centers[tail_mask]
tail_seeded   = seeded_counts[tail_mask].astype(float)
tail_clean    = counts[tail_mask].astype(float)

# log1p smoothing (adding 1) avoids log(0) without hardcoding a min count
# In the product/bone region both scans are similar → ratio ≈ 0 (flat)
# Where contaminant appears in seeded but not clean → ratio rises above 0
log_ratio = np.log10(tail_seeded + 1) - np.log10(tail_clean + 1)

# Use all bins — log1p smoothing means no bins need to be excluded
valid_centers = tail_centers
valid_counts  = log_ratio   # piecewise fit now operates on the ratio

n           = len(valid_centers)
min_segment = 5
residuals   = np.full(n, np.inf)

## At each point a straight line is fit to points left and right of the point. 
# The position where the two lines together fit best is the natural breakpoint in the data
for i in range(min_segment, n - min_segment):

    x_left   = valid_centers[:i]
    y_left   = valid_counts[:i]
    
    c_left   = np.polyfit(x_left, y_left, 1)
    res_left = np.sum((y_left - np.polyval(c_left, x_left)) ** 2)

    x_right   = valid_centers[i:]
    y_right   = valid_counts[i:]

    c_right   = np.polyfit(x_right, y_right, 1)
    res_right = np.sum((y_right - np.polyval(c_right, x_right)) ** 2)

    residuals[i] = res_left + res_right

best_idx          = int(np.argmin(residuals))
breakpoint_signal = float(valid_centers[best_idx])
threshold         = (breakpoint_signal // 100) * 100

print(f"\n── Piecewise linear breakpoint ────────────────────")
print(f"  breakpoint signal:    {breakpoint_signal:.0f}")
print(f"  suggested threshold:  {threshold:.0f} ")

#PSEUDOCODE
## Download the recipe excel sheet
## Find the recipe name that is closest to the contaminant threshold
## Compare contaminant properties to the recipe params
## If it doesn't work then find next closest. 
## Always start with the high recipe and then go to the low. 

recipe_df = load_recipe_library(RECIPE_PATH)
ranked = rank_recipes_by_proximity(recipe_df, threshold)
total_recipes = len(ranked)

for attempt, (_, recipe) in enumerate(ranked.iterrows(), start=1):
    recipe_name = recipe["Recipe_Name"]
    recipe_th = recipe["MainTh"]

    comp_df, label_arr, n_components = calculate_comp_values(recipe_th, seeded_data)
    matches_df = component_matches_recipe(comp_df, recipe)

    if len(matches_df) == n_components:
        print(f"    ✓ MATCH — {len(matches_df)} component(s) satisfy recipe criteria")
        print(matches_df.to_string(index=False))
        selected_recipe = {
            "recipe_name":        recipe_name,
            "recipe_row":         recipe,
            "matched_components": matches_df,
            "labeled_array":      label_arr,
            "n_components":       n_components,
            "attempts":           attempt,
        }
        break
    else:
        print(f"    All components did not statisfy recipe criteria "
            f"({len(comp_df)} components found, {len(matches_df)} qualified)")
        continue
        

# # ─────────── VISUALZING CONNECTED COMPONENTS ──────────────────────────────
# OUTPUT_PATH = "labeled_components_16bit.tiff"
# SEEDED_NEW_PATH = "seeded_data_16bit.tiff"
# # labeled_array is already the same shape as seeded_data
# # dtype: int32 — each voxel contains its component ID (0 = background)
# if os.path.exists(OUTPUT_PATH) & os.path.exists(SEEDED_NEW_PATH):
#     os.remove(OUTPUT_PATH)
#     os.remove(SEEDED_NEW_PATH)
#     print(f"Deleted existing file")

# tifffile.imwrite(OUTPUT_PATH, labeled_array.astype(np.uint16))
# tifffile.imwrite(SEEDED_NEW_PATH, seeded_data.astype(np.int16))

# # ─────────── PLOTTING USED FOR TESTING ONLY ───────────────────────────────
# fig, ax = plt.subplots(figsize=(12, 5))
# ax.plot(centers, counts, color="#1D9E75", lw=1.2, label="Clean (signal ≥ 0)")
# ax.plot(centers, seeded_counts, color="#EB700C", lw=1.2,
#         label="Seeded (signal ≥ 0)", alpha=0.8)
# ax.axvline(main_peak_signal, color="#1317E2", lw=1.5, ls="--",
#            label=f"Product peak: {main_peak_signal:.0f}")
# ax.axvline(threshold, color="#E2544A", lw=1.5, ls="--",
#            label=f"Contaminant threshold: {threshold:.0f}")
# ax.set_yscale("log")
# ax.set_ylim(bottom=0.5, top=8e6)
# ax.set_title("TEST — Histogram with negative values masked out", pad=10)
# ax.set_xlabel("Signal intensity")
# ax.set_ylabel("Voxel count")
# ax.legend(fontsize=9)

# # Add tick marks at every bin center location
# ax.set_xticks(centers)
# ax.tick_params(axis='x', which='major', length=4, width=0.5, labelsize=0)

# plt.tight_layout()
# plt.show()

