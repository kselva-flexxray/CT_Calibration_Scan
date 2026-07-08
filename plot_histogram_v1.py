import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import label, find_objects

# ── Configure ───────────────────────────────────────────────────────────
CLEAN_PATH          = "clean_vol_Diced_Beef_no_STD.tiff"
SEEDED_PATH         = "clean_vol_Diced_Beef_w_STD.tiff"
BINS                = 1024
NOISE_SIGNAL_MAX    = 300
BASELINE_SIGMA      = 3.0   # std devs above left tail mean to set baseline
SAFETY_MARGIN       = 0.05  # pull threshold down by this fraction as safety buffer
SMOOTH_SIGNAL_WIDTH = 25    # smoothing width in signal intensity units
SPARSE_COUNT        = 50    # voxels per bin below which product is considered ended
MIN_COUNTS          = 1
IQR_MULTIPLIER      = 0.75   # how many IQRs above local median to flag as outlier
IQR_WINDOW_BINS     = 8  

raw_data_plot  = False
diff_data_plot = True
# ────────────────────────────────────────────────────────────────────────

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

log_ratio = np.log10(tail_seeded + 1) - np.log10(tail_clean + 1)

# Use all bins — log1p smoothing means no bins need to be excluded
valid_centers = tail_centers
valid_counts  = log_ratio   # piecewise fit now operates on the ratio

n           = len(valid_centers)
min_segment = 4
residuals   = np.full(n, np.inf)

## At each point a straight line is fit to points left and right of the point. The position where the two lines together fit best is the natural breakpoint in the data
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

print(f"\n── Option C: Piecewise linear breakpoint ────────────────────")
print(f"  breakpoint signal:    {breakpoint_signal:.0f}")
print(f"  suggested threshold:  {threshold:.0f} ")

slag_mask = seeded_data >= threshold


# ── 3. Rolling IQR outlier detection ──────────────────────────────────────
full_tail_counts = tail_seeded.copy().astype(float)
log_full = np.where(full_tail_counts > 0,
                    np.log10(full_tail_counts),
                    -1.0)   # assign -1 to zero bins (below any real signal)

n_full       = len(tail_centers)
is_outlier   = np.zeros(n_full, dtype=bool)
local_median = np.zeros(n_full)
upper_fence  = np.zeros(n_full)

for i in range(n_full):
    lo = max(0, i - IQR_WINDOW_BINS)
    hi = min(n_full, i + IQR_WINDOW_BINS + 1)
    window_vals = log_full[lo:hi]

    # Only include non-zero bins in the IQR calculation
    real_vals = window_vals[window_vals > -1]

    if len(real_vals) < 3:
        # Not enough real signal in window — skip
        continue

    q25 = np.percentile(real_vals, 25)
    q75 = np.percentile(real_vals, 75)
    iqr = q75 - q25
    med = np.median(real_vals)

    local_median[i] = med
    upper_fence[i]  = med + IQR_MULTIPLIER * iqr

    if log_full[i] > upper_fence[i] and full_tail_counts[i] > 0:
        is_outlier[i] = True

outlier_signals = tail_centers[is_outlier]

print(f"\n── Rolling IQR outlier detection ──────────────────")
print(f"  outliers found: {is_outlier.sum()}")\

if len(outlier_signals) > 0:
    iqr_signal    = float(outlier_signals[0])
    threshold_e   = (iqr_signal // 100) * 100
    print(f"  first outlier signal: {iqr_signal:.0f}")
    print(f"  suggested threshold:  {threshold_e:.0f}")

# ─────────── PLOTTING USED FOR TESTING ONLY ───────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(centers, counts, color="#1D9E75", lw=1.2, label="Clean (signal ≥ 0)")
ax.plot(centers, seeded_counts, color="#EB700C", lw=1.2,
        label="Seeded (signal ≥ 0)", alpha=0.8)
ax.axvline(main_peak_signal, color="#1317E2", lw=1.5, ls="--",
           label=f"Product peak: {main_peak_signal:.0f}")
ax.axvline(threshold, color="#E2544A", lw=1.5, ls="--",
           label=f"Contaminant threshold: {threshold:.0f}")
ax.set_yscale("log")
ax.set_ylim(bottom=0.5, top=8e6)
ax.set_title("TEST — Histogram with negative values masked out", pad=10)
ax.set_xlabel("Signal intensity")
ax.set_ylabel("Voxel count")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

