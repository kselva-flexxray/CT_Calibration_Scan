import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# ── Configure these two paths ─────────────────────────────────────────────
CLEAN_PATH  = "clean_vol_Beef_Patties_no_STD.tiff"
SEEDED_PATH = "clean_vol_Beef_Patties_w_STD.tiff"
# ─────────────────────────────────────────────────────────────────────────

BINS        = 1024
SIGNAL_MIN  = 300
SMOOTH_SIGMA = 2      # gaussian smoothing before peak detection, increase if noisy
MIN_VOXELS   = 2      # minimum voxel count to be considered a real peak, not noise

def load_histogram(path, bins, shared_range):
    print(f"Loading {path} ...")
    data = tifffile.imread(path)
    data = data[data >= SIGNAL_MIN]
    counts, edges = np.histogram(data.ravel(), bins=bins, range=shared_range)
    print(f"  dtype: {data.dtype}  range: [{data.min()}, {data.max()}]")
    return counts, edges

# Compute shared range first so bin edges are identical
print("Reading data ...")
clean_data  = tifffile.imread(CLEAN_PATH)
seeded_data = tifffile.imread(SEEDED_PATH)

clean_data  = clean_data[clean_data   >= SIGNAL_MIN]
seeded_data = seeded_data[seeded_data >= SIGNAL_MIN]

shared_range = (min(clean_data.min(), seeded_data.min()),
                max(clean_data.max(), seeded_data.max()))

clean_counts,  clean_edges  = load_histogram(CLEAN_PATH,  BINS, shared_range)
seeded_counts, seeded_edges = load_histogram(SEEDED_PATH, BINS, shared_range)

centers = 0.5 * (clean_edges[:-1] + clean_edges[1:])

# ── Contaminant detection ─────────────────────────────────────────────────
diff          = seeded_counts.astype(float) - clean_counts.astype(float)
diff_smoothed = gaussian_filter1d(diff, sigma=SMOOTH_SIGMA)

# Find peaks in the positive residual only
positive_diff = np.clip(diff_smoothed, 0, None)
peaks, props  = find_peaks(
    positive_diff,
    height=MIN_VOXELS,      # ignore peaks smaller than MIN_VOXELS
    prominence=MIN_VOXELS,  # peak must stand out from surrounding baseline
    distance=5,             # minimum bins between peaks
)

# Filter out peaks that also exist in the clean scan (i.e. product peaks)
# A peak is "new" if the clean histogram at that bin is less than 10% of the seeded value
clean_smoothed = gaussian_filter1d(clean_counts.astype(float), sigma=SMOOTH_SIGMA)
new_peaks = [
    p for p in peaks
    if clean_smoothed[p] < 0.1 * positive_diff[p] or clean_counts[p] < MIN_VOXELS
]

print(f"\nDetected {len(new_peaks)} potential contaminant population(s):")
for p in new_peaks:
    # Estimate lower and upper bounds as the points where the peak drops to near zero
    left = p
    while left > 0 and diff_smoothed[left] > MIN_VOXELS / 2:
        left -= 1
    right = p
    while right < len(diff_smoothed) - 1 and diff_smoothed[right] > MIN_VOXELS / 2:
        right += 1
    print(f"  peak signal: {centers[p]:.0f}  "
          f"bounds: [{centers[left]:.0f}, {centers[right]:.0f}]  "
          f"voxels: {diff[p]:.0f}")

# Suggested threshold: lower bound of the first (lowest intensity) new peak
if new_peaks:
    first_peak = new_peaks[0]
    left = first_peak
    while left > 0 and diff_smoothed[left] > MIN_VOXELS / 2:
        left -= 1
    suggested_threshold = centers[left]
    print(f"\nSuggested minimum threshold: {suggested_threshold:.0f}")
else:
    suggested_threshold = None
    print("\nNo contaminant population detected — try reducing MIN_VOXELS or SMOOTH_SIGMA")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle("CT Calibration — Automated Contaminant Detection", fontsize=13, fontweight="bold")

# Panel 1: overlay, log scale
cn = clean_counts  / clean_counts.max()
sn = seeded_counts / seeded_counts.max()
ax1.fill_between(centers, cn, alpha=0.3, color="#1D9E75")
ax1.fill_between(centers, sn, alpha=0.3, color="#D85A30")
ax1.plot(centers, cn, color="#1D9E75", lw=1.5, label="Clean")
ax1.plot(centers, sn, color="#D85A30", lw=1.5, label="Seeded")
for p in new_peaks:
    ax1.axvline(centers[p], color="#EF9F27", lw=1.2, ls="--", alpha=0.8)
if suggested_threshold:
    ax1.axvline(suggested_threshold, color="#E24B4A", lw=1.5, ls=":",
                label=f"Suggested threshold: {suggested_threshold:.0f}")
ax1.set_yscale("log")
ax1.set_ylim(bottom=1e-6)
ax1.set_title("Overlay (normalised, log scale)")
ax1.set_xlabel("Signal intensity")
ax1.set_ylabel("Normalised voxel density (log)")
ax1.legend()

# Panel 2: difference, symlog, zoomed to right of product peak
product_peak_idx = np.argmax(clean_smoothed)
plot_min = centers[product_peak_idx] * 1.2   # start well past the product peak
mask = centers >= plot_min

ax2.fill_between(centers[mask], diff[mask], alpha=0.5, color="#E24B4A",
                 label="Residual (seeded − clean)")
ax2.plot(centers[mask], diff[mask], color="#E24B4A", lw=1.0)
ax2.axhline(0, color="#888780", lw=0.8, ls="--")

for p in new_peaks:
    if centers[p] >= plot_min:
        ax2.axvline(centers[p], color="#EF9F27", lw=1.5, ls="--")
        ax2.annotate(f"  {centers[p]:.0f}\n  {diff[p]:.0f} voxels",
                     xy=(centers[p], diff[p]),
                     xytext=(centers[p] + (centers[-1] - centers[0]) * 0.03, diff[p]),
                     fontsize=9, color="#633806",
                     arrowprops=dict(arrowstyle="->", color="#633806", lw=0.8))

if suggested_threshold and suggested_threshold >= plot_min:
    ax2.axvline(suggested_threshold, color="#E24B4A", lw=1.5, ls=":",
                label=f"Suggested threshold: {suggested_threshold:.0f}")

ax2.set_yscale("symlog", linthresh=10)
ax2.set_title("Seeded − Clean (zoomed past product peak, symlog scale)")
ax2.set_xlabel("Signal intensity")
ax2.set_ylabel("Voxel count difference (symlog)")
ax2.legend()

plt.tight_layout()
plt.savefig("ct_histogram_comparison.png", dpi=150, bbox_inches="tight")
print("\nPlot saved → ct_histogram_comparison.png")
plt.show()