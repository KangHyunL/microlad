"""
Three-axis latent refinement for 3D consistency
"""
import torch
from tqdm import tqdm


def three_axis_refinement(X, vae, refinement_K, device, desc="Three-axis refinement"):
    """
    Iterative latent refinement from three orthogonal views
    
    Enforces consistency by encoding-decoding slices along Z, Y, and X axes
    
    Args:
        X: [Z, 1, H, W] 3D volume as stack of 2D slices
        vae: VAE model for encode-decode
        refinement_K: Number of refinement iterations
        device: torch device
        desc: Progress bar description
    Returns:
        refined_X: [Z, 1, H, W] refined volume
    """
    Z, C, H, W = X.shape
    
    with torch.no_grad():
        for _ in tqdm(range(refinement_K), desc=desc):
            new_X = torch.zeros_like(X)
            
            # 1) Z-axis (depth slices)
            for z in range(Z):
                img_in = X[z:z+1] * 2 - 1  # [1, 1, H, W] in [-1, 1]
                mu, _ = vae.encode(img_in)
                dec = vae.decode(mu)[0, 0]  # [H, W]
                new_X[z, 0] += dec
            
            # 2) Y-axis (height slices)
            for y in range(H):
                sl = X[:, 0, y, :]  # [Z, W]
                img_in = sl.unsqueeze(0).unsqueeze(0) * 2 - 1  # [1, 1, Z, W]
                mu, _ = vae.encode(img_in)
                dec = vae.decode(mu)[0, 0]  # [Z, W]
                new_X[:, 0, y, :] += dec
            
            # 3) X-axis (width slices)
            for x in range(W):
                sl = X[:, 0, :, x]  # [Z, H]
                img_in = sl.unsqueeze(0).unsqueeze(0) * 2 - 1  # [1, 1, Z, H]
                mu, _ = vae.encode(img_in)
                dec = vae.decode(mu)[0, 0]  # [Z, H]
                new_X[:, 0, :, x] += dec
            
            # Average contributions from all three axes
            X = (new_X / 3.0).clamp(0, 1)
    
    return X

