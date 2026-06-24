import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# ── Configure ───────────────────────────────────────────────────────────
CLEAN_PATH          = "clean_vol_Beef_Patties_no_STD.tiff"
SEEDED_PATH         = "clean_vol_Beef_Patties_w_STD.tiff"
BINS                = 1024
NOISE_SIGNAL_MAX    = 200
BASELINE_SIGMA      = 3.0   # std devs above left tail mean to set baseline
SAFETY_MARGIN       = 0.05  # pull threshold down by this fraction as safety buffer
SMOOTH_SIGNAL_WIDTH = 25    # smoothing width in signal intensity units
SPARSE_COUNT        = 50    # voxels per bin below which product is considered ended

raw_data_plot  = False
diff_data_plot = True
# ────────────────────────────────────────────────────────────────────────

# ── Load ──────────────────────────────────────────────────────────────────
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

# ── FOR TESTING ONLY: mask out negative values and plot histogram ─────────
clean_data_masked  = clean_data[clean_data >= 0]
seeded_data_masked = seeded_data[seeded_data >= 0]

print(f"\n── Negative value masking (testing only) ────────────────────")
print(f"  clean  voxels removed:  {clean_data.size - clean_data_masked.size:,}")
print(f"  seeded voxels removed:  {seeded_data.size - seeded_data_masked.size:,}")

test_bins = 256
test_shared_range = (min(clean_data_masked.min(), seeded_data_masked.min()),
                      max(clean_data_masked.max(), seeded_data_masked.max()))
test_counts, test_edges = np.histogram(clean_data_masked.ravel(), bins=test_bins, range=test_shared_range)
test_seeded_counts, _   = np.histogram(seeded_data_masked.ravel(), bins=test_bins, range=test_shared_range)
test_centers = 0.5 * (test_edges[:-1] + test_edges[1:])

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(test_centers, test_counts, color="#1D9E75", lw=1.2, label="Clean (signal ≥ 0)")
ax.plot(test_centers, test_seeded_counts, color="#D85A30", lw=1.2,
        label="Seeded (signal ≥ 0)", alpha=0.8)
ax.set_yscale("log")
ax.set_ylim(bottom=0.5, top=7e6)
ax.set_title("TEST — Histogram with negative values masked out", pad=10)
ax.set_xlabel("Signal intensity")
ax.set_ylabel("Voxel count")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

# ── 1. Noise floor ────────────────────────────────────────────────────────
## Estimates of the noise floor, values of which will be later used to determine where the product signal begins
noise_voxels = clean_data[clean_data <= NOISE_SIGNAL_MAX].ravel().astype(float)
noise_mean   = noise_voxels.mean()
noise_std    = noise_voxels.std()
noise_floor  = noise_mean + 3 * noise_std

print(f"\n── Noise floor ──────────────────────────────")
print(f"  voxels sampled:   {len(noise_voxels):,}")
print(f"  mean:             {noise_mean:.2f}")
print(f"  std:              {noise_std:.2f}")
print(f"  floor (mean+3σ):  {noise_floor:.2f}")

# ── 2. Product peak ───────────────────────────────────────────────────────
## Uses a peak detection algorithm to find the location of the product peak
product_mask     = centers > noise_floor
peaks, _         = find_peaks(smooth[product_mask],
                               prominence=smooth[product_mask].max() * 0.05)
main_peak_idx    = np.where(product_mask)[0][peaks[np.argmax(smooth[product_mask][peaks])]]
main_peak_signal = centers[main_peak_idx]

print(f"\n── Product peak ─────────────────────────────")
print(f"  peak signal:  {main_peak_signal:.0f}")
print(f"  peak count:   {counts[main_peak_idx]:,}")

# ── 3. Tail from product peak ─────────────────────────────────────────────
## Right tail signal that will be used to search for contaminants 
tail_mask        = centers > main_peak_signal
tail_centers     = centers[tail_mask]
tail_diff_counts = seeded_counts[tail_mask].astype(float) - counts[tail_mask].astype(float)

# ── 4. Baseline from left tail (negative signal = pure reconstruction noise)
## 
left_tail_mask        = centers < 0
left_tail_diff_counts = (seeded_counts[left_tail_mask].astype(float)
                          - counts[left_tail_mask].astype(float))
baseline_mean = left_tail_diff_counts.mean()
baseline_std  = left_tail_diff_counts.std()
baseline      = baseline_mean + BASELINE_SIGMA * baseline_std

print(f"\n── Left tail baseline (signal < 0, seeded − clean) ──────────")
print(f"  bins in left tail:   {left_tail_mask.sum()}")
print(f"  mean diff:           {baseline_mean:.2f} voxels  (should be ≈ 0)")
print(f"  std diff:            {baseline_std:.2f} voxels")
print(f"  baseline (mean+{BASELINE_SIGMA:.0f}σ):  {baseline:.2f} voxels")

# ── 5. Find where product becomes sparse ──────────────────────────────────
sparse_idx    = np.where(counts[tail_mask] < SPARSE_COUNT)[0]
plot_left     = tail_centers[sparse_idx[0]] if len(sparse_idx) > 0 else main_peak_signal

print(f"\n── Sparse region ────────────────────────────")
print(f"  product tail ends at:       {plot_left:.0f}")

# ── 6. Rolling excess detection using left tail baseline ──────────────────
smooth_diff    = gaussian_filter1d(tail_diff_counts, sigma=sigma_bins)

# Only search above where product becomes sparse
search_mask    = tail_centers >= plot_left
search_diff    = smooth_diff[search_mask]
search_centers = tail_centers[search_mask]

excess_mask    = search_diff > baseline
excess_indices = np.where(excess_mask)[0]

contaminant_signal = None
threshold          = None

if len(excess_indices) > 0:
    # Require 2 consecutive bins to avoid single-bin spikes
    for i in range(len(excess_indices) - 1):
        if excess_indices[i + 1] == excess_indices[i] + 1:
            contaminant_signal = search_centers[excess_indices[i]]
            threshold          = contaminant_signal * (1 - SAFETY_MARGIN)
            break
    if contaminant_signal is None:
        contaminant_signal = search_centers[excess_indices[0]]
        threshold          = contaminant_signal * (1 - SAFETY_MARGIN)

    print(f"\n── Calibration result ───────────────────────")
    print(f"  first excess signal:  {contaminant_signal:.0f}")
    print(f"  suggested threshold:  {threshold:.0f}  (−{SAFETY_MARGIN*100:.0f}% safety margin)")
else:
    print("\n  No excess detected — try reducing BASELINE_SIGMA or SPARSE_COUNT")

# ── Plot ──────────────────────────────────────────────────────────────────
if raw_data_plot:
    fig = plt.figure(figsize=(12, 14))
    fig.suptitle("Scan Analysis: Noise Floor & Product Tail",
                 fontsize=13, fontweight="bold", y=1.0)

    gs     = fig.add_gridspec(2, 1, hspace=0.45)
    ax0    = fig.add_subplot(gs[0])
    gs_bot = gs[1].subgridspec(1, 2, wspace=0.3)
    ax1    = fig.add_subplot(gs_bot[0])
    ax2    = fig.add_subplot(gs_bot[1], sharey=ax1, sharex=ax1)

    noise_counts, noise_edges = np.histogram(noise_voxels, bins=200)
    noise_centers = 0.5 * (noise_edges[:-1] + noise_edges[1:])
    ax0.fill_between(noise_centers, noise_counts, alpha=0.4, color="#378ADD")
    ax0.plot(noise_centers, noise_counts, color="#378ADD", lw=1.2,
             label="Background voxels")
    ax0.axvline(noise_mean,             color="#1D9E75", lw=1.5, ls="-",
                label=f"Mean: {noise_mean:.2f}")
    ax0.axvline(noise_mean + noise_std, color="#EF9F27", lw=1.2, ls="--",
                label=f"Mean + 1σ: {noise_mean + noise_std:.2f}")
    ax0.axvline(noise_mean - noise_std, color="#EF9F27", lw=1.2, ls="--",
                label=f"Mean − 1σ: {noise_mean - noise_std:.2f}")
    ax0.axvline(noise_floor,            color="#E24B4A", lw=1.5, ls=":",
                label=f"Noise floor (mean+3σ): {noise_floor:.2f}")
    ax0.set_title(f"Background signal distribution (signal ≤ {NOISE_SIGNAL_MAX})", pad=10)
    ax0.set_xlabel("Signal intensity")
    ax0.set_ylabel("Voxel count")
    ax0.legend(fontsize=9)

    ax1.fill_between(tail_centers, counts[tail_mask], alpha=0.3, color="#1D9E75")
    ax1.plot(tail_centers, counts[tail_mask], color="#1D9E75", lw=1.0, label="Clean (tail)")
    ax1.set_yscale("log")
    ax1.set_xlim(left=main_peak_signal)
    ax1.set_ylim(bottom=0.5)
    ax1.set_title("Clean: right tail", pad=10)
    ax1.set_xlabel("Signal intensity")
    ax1.set_ylabel("Voxel count (log)")
    ax1.legend(fontsize=9)

    ax2.fill_between(tail_centers, seeded_counts[tail_mask], alpha=0.3, color="#D85A30")
    ax2.plot(tail_centers, seeded_counts[tail_mask], color="#D85A30", lw=1.0,
             label="Seeded (tail)")
    ax2.set_title("Seeded: right tail", pad=10)
    ax2.set_xlabel("Signal intensity")
    ax2.legend(fontsize=9)
    plt.setp(ax2.get_yticklabels(), visible=False)

    plt.savefig("ct_clean_analysis.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved → ct_clean_analysis.png")
    plt.show()

if diff_data_plot:
    pos_diff_counts = np.clip(tail_diff_counts, 0, None)
    neg_diff_counts = np.clip(tail_diff_counts, None, 0)

    fig, axes = plt.subplots(3, 1, figsize=(13, 14))
    fig.suptitle("CT Calibration — Tail Divergence & Threshold Derivation",
                 fontsize=13, fontweight="bold", y=1.01)

    # Panel 0: raw count tail overlay
    axes[0].plot(tail_centers, counts[tail_mask],        color="#1D9E75", lw=1.2, label="Clean")
    axes[0].plot(tail_centers, seeded_counts[tail_mask], color="#D85A30", lw=1.2,
                 label="Seeded", alpha=0.8)
    axes[0].axvline(plot_left, color="#888780", lw=1.0, ls="--",
                    label=f"Sparse start: {plot_left:.0f}")
    if threshold:
        axes[0].axvline(threshold, color="#E24B4A", lw=1.5, ls=":",
                        label=f"Suggested threshold: {threshold:.0f}")
    axes[0].set_yscale("log")
    axes[0].set_ylim(bottom=0.5)
    axes[0].set_xlim(left=main_peak_signal)
    axes[0].set_title("Raw voxel count tail overlay (log scale)", pad=10)
    axes[0].set_xlabel("Signal intensity")
    axes[0].set_ylabel("Voxel count (log)")
    axes[0].legend(fontsize=9)

    # Panel 1: bin-by-bin difference zoomed to sparse region
    axes[1].fill_between(tail_centers, pos_diff_counts, alpha=0.4, color="#D85A30",
                         label="Seeded > Clean")
    axes[1].fill_between(tail_centers, neg_diff_counts, alpha=0.4, color="#378ADD",
                         label="Clean > Seeded")
    axes[1].plot(tail_centers, tail_diff_counts, color="#888780", lw=0.8, alpha=0.6)
    axes[1].plot(tail_centers, smooth_diff, color="#333333", lw=1.5,
                 label=f"Smoothed diff ({SMOOTH_SIGNAL_WIDTH} signal units)")
    axes[1].axhline(0,        color="black",   lw=0.8, ls=":")
    axes[1].axhline( baseline, color="#EF9F27", lw=1.2, ls="--",
                    label=f"Baseline (mean+{BASELINE_SIGMA:.0f}σ): {baseline:.1f} voxels")
    axes[1].axhline(-baseline, color="#EF9F27", lw=1.2, ls="--")
    if contaminant_signal:
        axes[1].axvline(contaminant_signal, color="#E24B4A", lw=1.5, ls=":",
                        label=f"First excess: {contaminant_signal:.0f}")
    if threshold:
        axes[1].axvline(threshold, color="#633806", lw=1.5, ls="--",
                        label=f"Suggested threshold: {threshold:.0f}")
    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].set_xlim(left=plot_left)
    axes[1].set_title("Bin-by-bin difference — sparse region (symlog)", pad=10)
    axes[1].set_xlabel("Signal intensity")
    axes[1].set_ylabel("Voxel count difference (symlog)")
    axes[1].legend(fontsize=9)

    # Panel 2: zoomed into contaminant region
    if contaminant_signal:
        zoom_min  = max(plot_left, contaminant_signal - 200)
        zoom_max  = min(tail_centers[-1], contaminant_signal + 400)
        zoom_mask = (tail_centers >= zoom_min) & (tail_centers <= zoom_max)
        zoom_pos  = np.clip(tail_diff_counts[zoom_mask], 0, None)
        zoom_neg  = np.clip(tail_diff_counts[zoom_mask], None, 0)

        axes[2].fill_between(tail_centers[zoom_mask], zoom_pos,
                             alpha=0.4, color="#D85A30", label="Seeded > Clean")
        axes[2].fill_between(tail_centers[zoom_mask], zoom_neg,
                             alpha=0.4, color="#378ADD", label="Clean > Seeded")
        axes[2].plot(tail_centers[zoom_mask], smooth_diff[zoom_mask],
                     color="#333333", lw=1.5,
                     label=f"Smoothed diff ({SMOOTH_SIGNAL_WIDTH} signal units)")
        axes[2].axhline(baseline, color="#EF9F27", lw=1.2, ls="--",
                        label=f"Baseline: {baseline:.1f} voxels")
        axes[2].axhline(0, color="black", lw=0.8, ls=":")
        axes[2].axvline(contaminant_signal, color="#E24B4A", lw=1.5, ls=":",
                        label=f"First excess: {contaminant_signal:.0f}")
        axes[2].axvline(threshold, color="#633806", lw=1.5, ls="--",
                        label=f"Suggested threshold: {threshold:.0f}")
        axes[2].set_title("Zoomed — contaminant excess region", pad=10)
        axes[2].set_xlabel("Signal intensity")
        axes[2].set_ylabel("Voxel count difference")
        axes[2].legend(fontsize=9)
    else:
        axes[2].text(0.5, 0.5,
                     "No excess detected\nTry reducing BASELINE_SIGMA or SPARSE_COUNT",
                     transform=axes[2].transAxes, ha="center", va="center",
                     fontsize=12, color="gray")

    plt.tight_layout()
    plt.savefig("ct_calibration.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved → ct_calibration.png")
    plt.show()