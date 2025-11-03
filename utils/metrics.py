"""
Metrics computation for generated volumes
"""
import numpy as np


def compute_tpc_numpy(mask):
    """
    Compute TPC using FFT autocorrelation (NumPy version)
    
    Args:
        mask: [H, W] binary mask for single phase
    Returns:
        radial_profile: 1D array of S(r)
    """
    h, w = mask.shape
    n = h * w
    
    # FFT autocorrelation
    F = np.fft.fft2(mask)
    corr = np.fft.ifft2(F * np.conj(F))
    corr = np.real(corr) / n
    corr = np.fft.fftshift(corr)
    
    # Radial averaging
    cy, cx = h//2, w//2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    max_r = r.max()
    sum_r = np.bincount(r.ravel(), weights=corr.ravel(), minlength=max_r+1)
    count_r = np.bincount(r.ravel(), minlength=max_r+1)
    
    return sum_r / np.maximum(count_r, 1)


def compute_volume_tpc(volume, phases):
    """
    Compute TPC for a 3D volume by averaging over slices and axes
    
    Args:
        volume: [Z, Y, X] integer array with phase labels
        phases: List of phase indices
    Returns:
        tpc_dict: Dict mapping (axis, phase) -> TPC profile
    """
    Z, Y, X = volume.shape
    result = {}
    
    for axis_idx, axis_name, n_slices in [(0, 'z', Z), (1, 'y', Y), (2, 'x', X)]:
        for p in phases:
            tpcs = []
            
            for i in range(n_slices):
                # Extract slice
                if axis_idx == 0:
                    sl = volume[i, :, :]
                elif axis_idx == 1:
                    sl = volume[:, i, :]
                else:
                    sl = volume[:, :, i]
                
                # Binary mask for this phase
                mask = (sl == p).astype(np.float32)
                
                # Compute TPC
                tpc = compute_tpc_numpy(mask)
                tpcs.append(tpc)
            
            # Average over slices
            result[(axis_name, p)] = np.mean(np.stack(tpcs, axis=0), axis=0)
    
    return result

