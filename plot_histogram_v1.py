import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

# ── Configure ───────────────────────────────────────────────────────────
CLEAN_PATH       = "clean_vol_Beef_Patties_no_STD.tiff"
SEEDED_PATH      = "clean_vol_Beef_Patties_w_STD.tiff"
BINS             = 1024
NOISE_SIGNAL_MAX = 200

raw_data_plot = False
diff_data_plot = True
# ────────────────────────────────────────────────────────────────────────

# ── Load ─────────────────────────────────────────────────────────────────
print("Loading clean scan ...")
clean_data = tifffile.imread(CLEAN_PATH)

print("Loading seeded scan ...")
seeded_data = tifffile.imread(SEEDED_PATH)

# Use shared range so bin edges are identical for both histograms
shared_range = (min(clean_data.min(), seeded_data.min()),
                max(clean_data.max(), seeded_data.max()))

counts,        edges        = np.histogram(clean_data.ravel(),  bins=BINS, range=shared_range)
seeded_counts, seeded_edges = np.histogram(seeded_data.ravel(), bins=BINS, range=shared_range)
centers = 0.5 * (edges[:-1] + edges[1:])
smooth  = gaussian_filter1d(counts.astype(float), sigma=2)

# Normalise to voxel density (counts per total voxels)
clean_density  = counts.astype(float)  / counts.sum()
seeded_density = seeded_counts.astype(float) / seeded_counts.sum()

diff = seeded_density - clean_density

print(f"clean  total voxels: {clean_data.size:,}")
print(f"seeded total voxels: {seeded_data.size:,}")
print(f"clean  sum of counts: {counts.sum():,}")
print(f"seeded sum of counts: {seeded_counts.sum():,}")

# ── 1. Noise floor ────────────────────────────────────────────────────────
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
product_mask     = centers > noise_floor
peaks, _         = find_peaks(smooth[product_mask],
                               prominence=smooth[product_mask].max() * 0.05)
main_peak_idx    = np.where(product_mask)[0][peaks[np.argmax(smooth[product_mask][peaks])]]
main_peak_signal = centers[main_peak_idx]

print(f"\n── Product peak ─────────────────────────────")
print(f"  peak signal:  {main_peak_signal:.0f}")
print(f"  peak count:   {counts[main_peak_idx]:,}")

# ── Plot ──────────────────────────────────────────────────────────────────
if raw_data_plot: 
    fig = plt.figure(figsize=(12, 14))
    fig.suptitle("Clean Scan Analysis — Noise Floor & Product Tail",
                fontsize=13, fontweight="bold", y=1.0)  # push title above plots
    
    tail_mask = centers > main_peak_signal

    # Panel 0 takes the top half; panels 1 and 2 share the bottom half
    gs      = fig.add_gridspec(2, 1, hspace=0.45)
    ax0     = fig.add_subplot(gs[0])
    gs_bot  = gs[1].subgridspec(1, 2, wspace=0.3)
    ax1     = fig.add_subplot(gs_bot[0])
    ax2     = fig.add_subplot(gs_bot[1], sharey=ax1, sharex=ax1)  # shared axes

    # Panel 0: background signal distribution
    noise_counts, noise_edges = np.histogram(noise_voxels, bins=200)
    noise_centers = 0.5 * (noise_edges[:-1] + noise_edges[1:])   # fixed: was missing 0.5 *
    ax0.fill_between(noise_centers, noise_counts, alpha=0.4, color="#378ADD")
    ax0.plot(noise_centers, noise_counts, color="#378ADD", lw=1.2, label="Background voxels")
    ax0.axvline(noise_mean, color="#1D9E75", lw=1.5, ls="-",
                label=f"Mean: {noise_mean:.2f}")
    ax0.axvline(noise_mean + noise_std, color="#EF9F27", lw=1.2, ls="--",
                label=f"Mean + 1σ: {noise_mean + noise_std:.2f}")
    ax0.axvline(noise_mean - noise_std, color="#EF9F27", lw=1.2, ls="--",
                label=f"Mean − 1σ: {noise_mean - noise_std:.2f}")
    ax0.axvline(noise_floor, color="#E24B4A", lw=1.5, ls=":",
                label=f"Noise floor (mean+3σ): {noise_floor:.2f}")
    ax0.set_title(f"Background signal distribution (signal ≤ {NOISE_SIGNAL_MAX})", pad=10)
    ax0.set_xlabel("Signal intensity")
    ax0.set_ylabel("Voxel count")
    ax0.legend(fontsize=9)

    # Panels 1 & 2: clean and seeded tails on same x and y axes
    ax1.fill_between(centers[tail_mask], counts[tail_mask], alpha=0.3, color="#1D9E75")
    ax1.plot(centers[tail_mask], counts[tail_mask], color="#1D9E75", lw=1.0, label="Clean (tail)")
    ax1.set_yscale("log")
    ax1.set_xlim(left=main_peak_signal)
    ax1.set_ylim(bottom=0.5)
    ax1.set_title("Clean: right tail", pad=10)
    ax1.set_xlabel("Signal intensity")
    ax1.set_ylabel("Voxel count (log)")
    ax1.legend(fontsize=9)

    ax2.fill_between(centers[tail_mask], seeded_counts[tail_mask], alpha=0.3, color="#D85A30")
    ax2.plot(centers[tail_mask], seeded_counts[tail_mask], color="#D85A30", lw=1.0, label="Seeded (tail)")
    ax2.set_title("Seeded: right tail", pad=10)
    ax2.set_xlabel("Signal intensity")
    ax2.legend(fontsize=9)
    plt.setp(ax2.get_yticklabels(), visible=False)  # hide duplicate y labels

    plt.savefig("ct_clean_analysis.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved → ct_clean_analysis.png")
    plt.show()

if diff_data_plot: 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Tail Divergence — Seeded vs Clean", fontsize=13,
                fontweight="bold", y=1.01)
    
    tail_mask = centers > main_peak_signal

    clean_tail  = counts[tail_mask].astype(float)
    seeded_tail = seeded_counts[tail_mask].astype(float)
    tail_centers = centers[tail_mask]

    clean_tail   = clean_density[tail_mask]
    seeded_tail  = seeded_density[tail_mask]
    diff         = seeded_tail - clean_tail

    pos_diff = np.clip(diff, 0, None)
    neg_diff = np.clip(diff, None, 0)

    # diff = seeded_tail - clean_tail
    # # Split into positive and negative for separate colouring
    # pos_diff = np.clip(diff, 0, None)
    # neg_diff = np.clip(diff, None, 0)

    # Panel 1: overlay of both tails for context
    ax1.plot(tail_centers, clean_tail,  color="#1D9E75", lw=1.2, label="Clean")
    ax1.plot(tail_centers, seeded_tail, color="#D85A30", lw=1.2, label="Seeded", alpha=0.8)
    ax1.set_yscale("log")
    ax1.set_ylim(bottom=0.5)
    ax1.set_ylabel("Voxel count (log)")
    ax1.set_title("Clean vs Seeded tails", pad=10)
    ax1.legend(fontsize=9)
    
    # Panel 2: difference, positive and negative separately
    ax2.fill_between(tail_centers, pos_diff, alpha=0.4, color="#D85A30",
                    label="Seeded > Clean (contaminant candidate)")
    ax2.fill_between(tail_centers, neg_diff, alpha=0.4, color="#378ADD",
                    label="Clean > Seeded (product shift)")
    ax2.plot(tail_centers, diff, color="#888780", lw=0.8, alpha=0.6)
    ax2.axhline(0, color="black", lw=0.8, ls="--")
    ax2.set_yscale("symlog", linthresh=1)
    ax2.set_ylabel("Voxel count difference (symlog)")
    ax2.set_xlabel("Signal intensity")
    ax2.set_title("Seeded − Clean difference", pad=10)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("ct_tail_divergence.png", dpi=150, bbox_inches="tight")
    print("Plot saved → ct_tail_divergence.png")
    plt.show()