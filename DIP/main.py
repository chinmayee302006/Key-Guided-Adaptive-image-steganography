"""
Main execution script for:
Key-Guided Adaptive Multi-Bit LSB Image Steganography
Using Local Entropy and Channel Optimization

Runs all three methods, compares results, generates plots and research outputs.
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from steganography import (
    compute_entropy_map, adaptive_bit_allocation,
    embed_basic_lsb, extract_basic_lsb,
    embed_random_lsb, extract_random_lsb,
    embed_adaptive, extract_adaptive,
    compute_mse, compute_psnr,
    compute_capacity_bpp, compute_max_capacity_bpp,
    get_pixel_bits
)

# ============================================================
# CONFIGURATION
# ============================================================
SECRET_KEY = 42
BLOCK_SIZE = 8
LOW_THRESH = 1.5
HIGH_THRESH = 3.5

SECRET_MESSAGE = (
    "This is a secret message for steganography research. "
    "The quick brown fox jumps over the lazy dog. "
    "Image steganography hides data within digital images "
    "by modifying pixel values in an imperceptible manner. "
    "This implementation uses entropy-based adaptive bit allocation "
    "combined with key-guided randomization for enhanced security."
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def load_or_create_cover_image(path=None):
    """Load a cover image or create a synthetic test image."""
    if path and os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            # Resize to ensure dimensions are multiples of block size
            h, w = img.shape[:2]
            h = (h // BLOCK_SIZE) * BLOCK_SIZE
            w = (w // BLOCK_SIZE) * BLOCK_SIZE
            img = cv2.resize(img, (w, h))
            print(f"[INFO] Loaded cover image: {path} ({w}x{h})")
            return img

    # Create synthetic test image with varied textures
    print("[INFO] Creating synthetic cover image (512x512)...")
    h, w = 512, 512
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Smooth gradient sky (low entropy)
    for r in range(h // 3):
        ratio = r / (h // 3)
        img[r, :, 0] = int(200 - 60 * ratio)  # B
        img[r, :, 1] = int(180 - 40 * ratio)  # G
        img[r, :, 2] = int(120 + 50 * ratio)  # R

    # Medium texture region (medium entropy)
    for r in range(h // 3, 2 * h // 3):
        for c in range(w):
            base_g = 80 + int(40 * np.sin(c / 15.0))
            base_r = 50 + int(30 * np.cos(r / 12.0))
            img[r, c] = [30, base_g, base_r]

    # Complex texture (high entropy)
    rng = np.random.RandomState(123)
    noise = rng.randint(0, 256, (h // 3, w, 3), dtype=np.uint8)
    base = np.zeros((h // 3, w, 3), dtype=np.uint8)
    for r in range(h // 3):
        for c in range(w):
            base[r, c] = [50 + c % 80, 100 + r % 60, 60 + (r + c) % 90]
    img[2 * h // 3:, :] = cv2.addWeighted(base, 0.6, noise, 0.4, 0)

    cv2.imwrite(os.path.join(RESULTS_DIR, "cover_synthetic.png"), img)
    return img


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_entropy_and_allocation(cover, block_size=8):
    """Visualize entropy map and adaptive bit allocation."""
    entropy_map = compute_entropy_map(cover, block_size)
    bit_map = adaptive_bit_allocation(entropy_map, LOW_THRESH, HIGH_THRESH)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Original image
    axes[0].imshow(cv2.cvtColor(cover, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Cover Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Entropy map
    im1 = axes[1].imshow(entropy_map, cmap='hot', interpolation='nearest')
    axes[1].set_title("Block Entropy Map", fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Bit allocation map
    im2 = axes[2].imshow(bit_map, cmap='RdYlGn', interpolation='nearest',
                         vmin=1, vmax=3)
    axes[2].set_title("Adaptive Bit Allocation (1-3 bits)", fontsize=14,
                      fontweight='bold')
    cbar = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04,
                        ticks=[1, 2, 3])
    cbar.ax.set_yticklabels(['1 bit\n(smooth)', '2 bits\n(medium)',
                             '3 bits\n(complex)'])

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "entropy_allocation.png"), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("[SAVED] entropy_allocation.png")


def plot_stego_comparison(cover, stegos, method_names):
    """Side-by-side comparison of cover and stego images."""
    n = len(stegos) + 1
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    axes[0].imshow(cv2.cvtColor(cover, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Cover", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    for i, (stego, name) in enumerate(zip(stegos, method_names)):
        axes[i + 1].imshow(cv2.cvtColor(stego, cv2.COLOR_BGR2RGB))
        psnr = compute_psnr(cover, stego)
        axes[i + 1].set_title(f"{name}\nPSNR: {psnr:.2f} dB", fontsize=12,
                              fontweight='bold')
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "stego_comparison.png"), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("[SAVED] stego_comparison.png")


def plot_difference_images(cover, stegos, method_names):
    """Show amplified difference between cover and stego images."""
    fig, axes = plt.subplots(1, len(stegos), figsize=(6 * len(stegos), 5))
    if len(stegos) == 1:
        axes = [axes]

    for i, (stego, name) in enumerate(zip(stegos, method_names)):
        diff = cv2.absdiff(cover, stego).astype(np.float64)
        diff_amp = np.clip(diff * 20, 0, 255).astype(np.uint8)
        axes[i].imshow(cv2.cvtColor(diff_amp, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Difference (20x): {name}", fontsize=12,
                          fontweight='bold')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "difference_images.png"), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("[SAVED] difference_images.png")


def plot_histogram_comparison(cover, stegos, method_names):
    """Compare histograms of cover vs stego images."""
    channels = ['Blue', 'Green', 'Red']
    colors_plt = ['#2196F3', '#4CAF50', '#F44336']

    fig, axes = plt.subplots(len(stegos), 3, figsize=(18, 5 * len(stegos)))
    if len(stegos) == 1:
        axes = axes.reshape(1, -1)

    for i, (stego, name) in enumerate(zip(stegos, method_names)):
        for c in range(3):
            hist_cover = cv2.calcHist([cover], [c], None, [256], [0, 256])
            hist_stego = cv2.calcHist([stego], [c], None, [256], [0, 256])

            axes[i, c].plot(hist_cover, color=colors_plt[c], alpha=0.7,
                            linewidth=1.5, label='Cover')
            axes[i, c].plot(hist_stego, color='black', alpha=0.5,
                            linewidth=1.5, linestyle='--', label='Stego')
            axes[i, c].set_title(f"{name} — {channels[c]} Channel",
                                 fontsize=11, fontweight='bold')
            axes[i, c].legend()
            axes[i, c].set_xlim([0, 256])

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "histogram_comparison.png"), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("[SAVED] histogram_comparison.png")


def plot_metrics_comparison(metrics):
    """Bar charts comparing PSNR, MSE, and Capacity."""
    names = [m['name'] for m in metrics]
    psnrs = [m['psnr'] for m in metrics]
    mses = [m['mse'] for m in metrics]
    caps = [m['max_bpp'] for m in metrics]

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # PSNR comparison
    bars = axes[0].bar(names, psnrs, color=colors, edgecolor='white',
                       linewidth=2, width=0.6)
    axes[0].set_title("PSNR Comparison (dB)", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("PSNR (dB)", fontsize=12)
    for bar, val in zip(bars, psnrs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')
    axes[0].set_ylim(0, max(psnrs) * 1.15)

    # MSE comparison
    bars = axes[1].bar(names, mses, color=colors, edgecolor='white',
                       linewidth=2, width=0.6)
    axes[1].set_title("MSE Comparison", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("MSE", fontsize=12)
    for bar, val in zip(bars, mses):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                     f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')

    # Capacity comparison
    bars = axes[2].bar(names, caps, color=colors, edgecolor='white',
                       linewidth=2, width=0.6)
    axes[2].set_title("Max Capacity (bits/pixel)", fontsize=14, fontweight='bold')
    axes[2].set_ylabel("Bits per Pixel", fontsize=12)
    for bar, val in zip(bars, caps):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "metrics_comparison.png"), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("[SAVED] metrics_comparison.png")


def plot_entropy_distribution(cover, block_size=8):
    """Plot distribution of block entropy values."""
    entropy_map = compute_entropy_map(cover, block_size)
    entropies = entropy_map.flatten()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(entropies, bins=50, color='#45B7D1', edgecolor='white',
            linewidth=1.2, alpha=0.85)
    ax.axvline(x=LOW_THRESH, color='#FF6B6B', linewidth=2, linestyle='--',
               label=f'Low threshold ({LOW_THRESH})')
    ax.axvline(x=HIGH_THRESH, color='#4CAF50', linewidth=2, linestyle='--',
               label=f'High threshold ({HIGH_THRESH})')
    ax.set_title("Distribution of Block Entropy Values", fontsize=14,
                 fontweight='bold')
    ax.set_xlabel("Entropy (bits)", fontsize=12)
    ax.set_ylabel("Number of Blocks", fontsize=12)
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "entropy_distribution.png"), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("[SAVED] entropy_distribution.png")


# ============================================================
# MAIN EXECUTION
# ============================================================

def run_analysis(cover_image_path=None):
    """Run complete steganography analysis."""
    ensure_results_dir()
    print("=" * 70)
    print("  Key-Guided Adaptive Multi-Bit LSB Image Steganography")
    print("  Using Local Entropy and Channel Optimization")
    print("=" * 70)

    # Load cover image
    cover = load_or_create_cover_image(cover_image_path)
    h, w = cover.shape[:2]
    print(f"\n[CONFIG] Image size: {w}x{h} ({w*h} pixels)")
    print(f"[CONFIG] Block size: {BLOCK_SIZE}")
    print(f"[CONFIG] Secret key: {SECRET_KEY}")
    print(f"[CONFIG] Message length: {len(SECRET_MESSAGE)} chars "
          f"({len(SECRET_MESSAGE)*8} bits)")

    # Save cover image
    cv2.imwrite(os.path.join(RESULTS_DIR, "cover_image.png"), cover)

    # Entropy analysis
    print("\n--- Entropy Analysis ---")
    entropy_map = compute_entropy_map(cover, BLOCK_SIZE)
    bit_map = adaptive_bit_allocation(entropy_map, LOW_THRESH, HIGH_THRESH)
    print(f"  Entropy range: [{entropy_map.min():.3f}, {entropy_map.max():.3f}]")
    print(f"  Mean entropy: {entropy_map.mean():.3f}")
    n1 = np.sum(bit_map == 1)
    n2 = np.sum(bit_map == 2)
    n3 = np.sum(bit_map == 3)
    total_blocks = bit_map.size
    print(f"  1-bit blocks: {n1} ({100*n1/total_blocks:.1f}%)")
    print(f"  2-bit blocks: {n2} ({100*n2/total_blocks:.1f}%)")
    print(f"  3-bit blocks: {n3} ({100*n3/total_blocks:.1f}%)")

    # ---- METHOD 1: Basic LSB ----
    print("\n" + "=" * 50)
    print("  METHOD 1: Basic LSB (Sequential, 1-bit)")
    print("=" * 50)
    stego_basic, used_basic, cap_basic = embed_basic_lsb(cover, SECRET_MESSAGE)
    extracted_basic = extract_basic_lsb(stego_basic)
    mse_basic = compute_mse(cover, stego_basic)
    psnr_basic = compute_psnr(cover, stego_basic)
    bpp_basic = compute_capacity_bpp(used_basic, cover.shape)
    max_bpp_basic = compute_max_capacity_bpp(cap_basic, cover.shape)

    print(f"  Bits embedded: {used_basic}")
    print(f"  MSE:  {mse_basic:.6f}")
    print(f"  PSNR: {psnr_basic:.2f} dB")
    print(f"  Used BPP: {bpp_basic:.4f}")
    print(f"  Max BPP:  {max_bpp_basic:.4f}")
    print(f"  Extraction OK: {extracted_basic == SECRET_MESSAGE}")
    cv2.imwrite(os.path.join(RESULTS_DIR, "stego_basic.png"), stego_basic)

    # ---- METHOD 2: Random LSB ----
    print("\n" + "=" * 50)
    print("  METHOD 2: Random LSB (Key-Randomized, 1-bit)")
    print("=" * 50)
    stego_random, used_random, cap_random = embed_random_lsb(
        cover, SECRET_MESSAGE, SECRET_KEY)
    extracted_random = extract_random_lsb(stego_random, SECRET_KEY)
    mse_random = compute_mse(cover, stego_random)
    psnr_random = compute_psnr(cover, stego_random)
    bpp_random = compute_capacity_bpp(used_random, cover.shape)
    max_bpp_random = compute_max_capacity_bpp(cap_random, cover.shape)

    print(f"  Bits embedded: {used_random}")
    print(f"  MSE:  {mse_random:.6f}")
    print(f"  PSNR: {psnr_random:.2f} dB")
    print(f"  Used BPP: {bpp_random:.4f}")
    print(f"  Max BPP:  {max_bpp_random:.4f}")
    print(f"  Extraction OK: {extracted_random == SECRET_MESSAGE}")
    cv2.imwrite(os.path.join(RESULTS_DIR, "stego_random.png"), stego_random)

    # ---- METHOD 3: Proposed Adaptive ----
    print("\n" + "=" * 50)
    print("  METHOD 3: Proposed Adaptive (Entropy + Channel Opt.)")
    print("=" * 50)
    stego_adaptive, used_adaptive, cap_adaptive = embed_adaptive(
        cover, SECRET_MESSAGE, SECRET_KEY, BLOCK_SIZE, LOW_THRESH, HIGH_THRESH)
    extracted_adaptive = extract_adaptive(
        stego_adaptive, SECRET_KEY, BLOCK_SIZE, LOW_THRESH, HIGH_THRESH)
    mse_adaptive = compute_mse(cover, stego_adaptive)
    psnr_adaptive = compute_psnr(cover, stego_adaptive)
    bpp_adaptive = compute_capacity_bpp(used_adaptive, cover.shape)
    max_bpp_adaptive = compute_max_capacity_bpp(cap_adaptive, cover.shape)

    print(f"  Bits embedded: {used_adaptive}")
    print(f"  MSE:  {mse_adaptive:.6f}")
    print(f"  PSNR: {psnr_adaptive:.2f} dB")
    print(f"  Used BPP: {bpp_adaptive:.4f}")
    print(f"  Max BPP:  {max_bpp_adaptive:.4f}")
    print(f"  Extraction OK: {extracted_adaptive == SECRET_MESSAGE}")
    cv2.imwrite(os.path.join(RESULTS_DIR, "stego_adaptive.png"), stego_adaptive)

    # ---- VISUALIZATION ----
    print("\n--- Generating Visualizations ---")
    method_names = ["Basic LSB", "Random LSB", "Proposed"]
    stegos = [stego_basic, stego_random, stego_adaptive]

    metrics = [
        {'name': 'Basic LSB', 'psnr': psnr_basic, 'mse': mse_basic,
         'used_bpp': bpp_basic, 'max_bpp': max_bpp_basic},
        {'name': 'Random LSB', 'psnr': psnr_random, 'mse': mse_random,
         'used_bpp': bpp_random, 'max_bpp': max_bpp_random},
        {'name': 'Proposed', 'psnr': psnr_adaptive, 'mse': mse_adaptive,
         'used_bpp': bpp_adaptive, 'max_bpp': max_bpp_adaptive},
    ]

    plot_entropy_and_allocation(cover, BLOCK_SIZE)
    plot_entropy_distribution(cover, BLOCK_SIZE)
    plot_stego_comparison(cover, stegos, method_names)
    plot_difference_images(cover, stegos, method_names)
    plot_histogram_comparison(cover, stegos, method_names)
    plot_metrics_comparison(metrics)

    # ---- SUMMARY TABLE ----
    print("\n" + "=" * 70)
    print("  COMPARATIVE RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<18} {'PSNR (dB)':>10} {'MSE':>12} {'Used BPP':>10} "
          f"{'Max BPP':>10} {'Extract OK':>12}")
    for m, extracted in zip(metrics, [extracted_basic, extracted_random, extracted_adaptive]):
        ok_str = "YES" if extracted == SECRET_MESSAGE else "NO"
        print(f"{m['name']:<18} {m['psnr']:>10.2f} {m['mse']:>12.6f} "
              f"{m['used_bpp']:>10.4f} {m['max_bpp']:>10.4f} {ok_str:>12}")
    print("-" * 72)

    print(f"\n[INFO] All results saved to: {RESULTS_DIR}")
    print("[DONE] Analysis complete!")

    return metrics


if __name__ == "__main__":
    # Check for command-line image path argument
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_analysis(img_path)
