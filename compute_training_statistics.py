"""
Compute TPC, VF, and SA statistics from training images
"""
import os
import argparse
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def compute_tpc(mask):
    """Compute TPC autocorrelation with radial averaging"""
    h, w = mask.shape
    n = h * w
    
    F = np.fft.fft2(mask)
    corr = np.real(np.fft.ifft2(F * np.conj(F))) / n
    corr = np.fft.fftshift(corr)
    
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    max_r = r.max()
    sum_r = np.bincount(r.ravel(), weights=corr.ravel(), minlength=max_r + 1)
    count_r = np.bincount(r.ravel(), minlength=max_r + 1)
    
    return sum_r / np.maximum(count_r, 1)


def compute_stats(img, phases):
    """Compute TPC, VF, and SA for one image"""
    H, W = img.shape
    stats = {}
    
    for p in phases:
        mask = (img == p).astype(np.float32)
        
        # TPC
        stats[p] = {
            'tpc': compute_tpc(mask),
            'vf': float(mask.mean()),
            'sa': float((np.abs(gaussian_filter(mask, 1.0)[1:, :] - gaussian_filter(mask, 1.0)[:-1, :]).sum() +
                        np.abs(gaussian_filter(mask, 1.0)[:, 1:] - gaussian_filter(mask, 1.0)[:, :-1]).sum()) / (H*W))
        }
    
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    
    # Find images
    files = []
    for ext in ['png', 'jpg', 'tif', 'tiff', 'bmp']:
        files.extend(glob(os.path.join(args.input_dir, f'*.{ext}')))
        files.extend(glob(os.path.join(args.input_dir, f'**/*.{ext}'), recursive=True))
    files = sorted(set(files))
    
    print(f"Found {len(files)} images")
    
    first = np.array(Image.open(files[0]).convert('L'))
    phases = sorted(np.unique(first))
    print(f"Phases: {phases}")
    
    # Collect
    all_tpc = {p: [] for p in phases}
    all_vf = {p: [] for p in phases}
    all_sa = {p: [] for p in phases}
    
    for path in tqdm(files, desc="Processing"):
        img = np.array(Image.open(path).convert('L'))
        stats = compute_stats(img, phases)
        
        for p in phases:
            all_tpc[p].append(stats[p]['tpc'])
            all_vf[p].append(stats[p]['vf'])
            all_sa[p].append(stats[p]['sa'])
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    save_dict = {}
    
    for p in phases:
        tpc_array = np.stack(all_tpc[p], axis=0)
        save_dict[f"S{p}{p}_mean"] = tpc_array.mean(axis=0)
        save_dict[f"S{p}{p}_std"] = tpc_array.std(axis=0)
        save_dict[f"VF{p}"] = all_vf[p]
        save_dict[f"SA{p}"] = all_sa[p]
        
        print(f"\nPhase {p}: VF={np.mean(all_vf[p]):.4f}±{np.std(all_vf[p]):.4f}, SA={np.mean(all_sa[p]):.4f}±{np.std(all_sa[p]):.4f}")
    
    np.savez(os.path.join(args.output_dir, 'autocorr_periodic_mean_std.npz'), **save_dict)
    print(f"\nSaved: {args.output_dir}/autocorr_periodic_mean_std.npz")
    
    # Plot
    for p in phases:
        mean_tpc = save_dict[f"S{p}{p}_mean"]
        std_tpc = save_dict[f"S{p}{p}_std"]
        plt.figure(figsize=(6, 4))
        plt.plot(mean_tpc, linewidth=2)
        plt.fill_between(range(len(mean_tpc)), mean_tpc - std_tpc, mean_tpc + std_tpc, alpha=0.3)
        plt.xlabel('r (pixels)')
        plt.ylabel(f'S_{p}{p}(r)')
        plt.title(f'Phase {p} TPC')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'tpc_phase{p}.png'), dpi=150)
        plt.close()


if __name__ == '__main__':
    main()

