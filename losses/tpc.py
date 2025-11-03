"""
Two-Point Correlation Function (TPC) computation and loss
"""
import torch
import torch.nn.functional as F


def setup_tpc_bins(H, W, device):
    """
    Precompute radial bins for TPC calculation
    
    Args:
        H, W: Image dimensions
        device: torch device
    Returns:
        bin_mat: [nbins, H*W] one-hot encoding of radius bins
        bin_counts: [nbins, 1] number of pixels per bin
    """
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    
    r = ((yy - H//2)**2 + (xx - W//2)**2).sqrt().round().long().view(-1)
    nbins = int(r.max().item()) + 1
    
    bin_mat = F.one_hot(r, num_classes=nbins).float().transpose(0,1)
    bin_counts = bin_mat.sum(dim=1, keepdim=True)
    
    return bin_mat, bin_counts


def compute_tpc_torch(mask, bin_mat, bin_counts):
    """
    Compute TPC via FFT autocorrelation and radial averaging
    
    Args:
        mask: [H, W] binary or soft mask
        bin_mat: [nbins, H*W] one-hot radius bins
        bin_counts: [nbins, 1] pixels per bin
    Returns:
        tpc: [nbins] radial TPC profile
    """
    H, W = mask.shape
    
    # FFT-based autocorrelation
    Ff   = torch.fft.fft2(mask)
    corr = torch.fft.ifft2(Ff * torch.conj(Ff)) / (H * W)
    corr = torch.real(torch.fft.fftshift(corr))
    
    # Radial averaging
    corr_flat = corr.reshape(-1)
    tpc = bin_mat @ corr_flat
    tpc = tpc / bin_counts.squeeze(1)
    
    return tpc


def compute_tpc_loss_ste(masks_p, phases, tpc_targets, bin_mat, bin_counts, device):
    """
    TPC loss using Straight-Through Estimator (STE)
    
    Forward: Uses hard (argmax) masks for exact TPC matching
    Backward: Gradients flow through soft masks
    
    Args:
        masks_p: [P, H, W] soft probability masks per phase
        phases: List of phase indices
        tpc_targets: Dict mapping phase -> target TPC profile
        bin_mat: Precomputed radial bins
        bin_counts: Precomputed bin counts
        device: torch device
    Returns:
        loss_tpc: Scalar TPC matching loss
    """
    loss_tpc = torch.tensor(0.0, device=device)
    
    # Get hard assignment via argmax
    hard_assignment = masks_p.argmax(dim=0)  # [H, W] indices
    
    for pi, p in enumerate(phases):
        # Hard mask (forward)
        hard_mask = (hard_assignment == pi).float()  # [H, W] binary
        
        # Soft mask (backward)
        soft_mask = masks_p[pi]  # [H, W] probabilities
        
        # Straight-through estimator
        mask_ste = hard_mask - soft_mask.detach() + soft_mask
        
        # Compute TPC on STE mask
        tpc_pred = compute_tpc_torch(mask_ste, bin_mat, bin_counts)
        
        # Target TPC
        tgt = torch.tensor(tpc_targets[p], device=device, dtype=tpc_pred.dtype)
        L = min(tpc_pred.shape[0], tgt.shape[0])
        
        # Accumulate MSE
        loss_tpc = loss_tpc + F.mse_loss(tpc_pred[:L], tgt[:L])
    
    # Average over phases
    return loss_tpc / len(phases)


def setup_tpc_bins(H, W, device):
    """
    Precompute radial bins for TPC calculation
    
    Args:
        H, W: Image dimensions
        device: torch device
    Returns:
        bin_mat: [nbins, H*W] one-hot encoding of radius bins
        bin_counts: [nbins, 1] number of pixels per bin
    """
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    
    # Distance from center
    r = ((yy - H//2)**2 + (xx - W//2)**2).sqrt().round().long().view(-1)
    nbins = int(r.max().item()) + 1
    
    # One-hot encoding
    bin_mat = F.one_hot(r, num_classes=nbins).float().transpose(0,1)  # [nbins, H*W]
    bin_counts = bin_mat.sum(dim=1, keepdim=True)  # [nbins, 1]
    
    return bin_mat, bin_counts

