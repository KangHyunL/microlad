"""
Surface area computation and matching
"""
import torch
import torch.nn.functional as F


def _make_gaussian_kernel(kernel_size, sigma, device):
    """
    Create Gaussian kernel for smoothing
    
    Args:
        kernel_size: Kernel size (odd number)
        sigma: Standard deviation
        device: torch device
    Returns:
        kernel: [1, 1, ks, ks] Gaussian kernel
    """
    ax = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


def compute_relative_SA_torch(recon, phases, kernel_size=7, sigma=1.0, beta=50.0):
    """
    Compute relative surface area for each phase using total variation
    
    Args:
        recon: [B, 1, H, W] microstructure in [0,1]
        phases: List of phase indices
        kernel_size: Gaussian kernel size for smoothing
        sigma: Gaussian standard deviation
        beta: Softmax temperature
    Returns:
        sa: [P] relative surface area per phase
    """
    B, _, H, W = recon.shape
    device, dtype = recon.device, recon.dtype
    P = len(phases)
    
    # Soft phase assignment
    levels = torch.linspace(0.0, 1.0, steps=P, device=device, dtype=dtype)
    x = recon.expand(B, P, H, W)
    l = levels.view(1, P, 1, 1)
    dist = torch.abs(x - l)
    masks = F.softmax(-beta * dist.view(B, P, -1), dim=1).view(B, P, H, W)
    
    # Gaussian smoothing
    gk = _make_gaussian_kernel(kernel_size, sigma, device).repeat(P, 1, 1, 1)
    M_s = F.conv2d(masks, weight=gk, padding=kernel_size//2, groups=P)
    
    # Total variation (surface area proxy)
    tv_h = torch.abs(M_s[:, :, 1:, :] - M_s[:, :, :-1, :]).sum(dim=(2,3))  # [B, P]
    tv_w = torch.abs(M_s[:, :, :, 1:] - M_s[:, :, :, :-1]).sum(dim=(2,3))  # [B, P]
    sa = (tv_h + tv_w) / (H * W)
    
    return sa.mean(dim=0)  # [P]


def compute_sa_loss(decoded, sa_targets, phases, device):
    """
    Compute surface area matching loss
    
    Args:
        decoded: [B, 1, H, W] microstructure
        sa_targets: Dict mapping phase -> target SA value
        phases: List of phase indices
        device: torch device
    Returns:
        loss_sa: Scalar SA loss
    """
    rel_sa = compute_relative_SA_torch(decoded, phases, kernel_size=7, sigma=1.0)
    tgt_sa = torch.tensor([sa_targets[p] for p in phases], device=device)
    return F.mse_loss(rel_sa, tgt_sa)

