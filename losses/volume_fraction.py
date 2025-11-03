"""
Volume fraction matching via moment matching
"""
import torch
import torch.nn.functional as F


def compute_vf_loss(decoded, vf0, vf05, vf1, w_m1, w_m2, device):
    """
    Compute volume fraction loss via moment matching
    
    Matches first two moments of the distribution to target VF
    
    Args:
        decoded: [B, 1, H, W] generated microstructure in [0,1]
        vf0, vf05, vf1: Target volume fractions for phases {0, 0.5, 1}
        w_m1, w_m2: Weights for moment 1 and moment 2
        device: torch device
    Returns:
        loss_m1, loss_m2: Moment matching losses
        m1: Actual first moment (for monitoring)
    """
    B = decoded.shape[0]
    
    # Compute empirical moments
    m1 = decoded.view(B, -1).mean(dim=1)       # E[x]
    m2 = (decoded**2).view(B, -1).mean(dim=1)  # E[x²]
    
    # Compute target moments from volume fractions
    # E[x] = 0.0*vf0 + 0.5*vf05 + 1.0*vf1
    t_mean = 0.0*vf0 + 0.5*vf05 + 1.0*vf1
    
    # E[x²] = 0.0²*vf0 + 0.5²*vf05 + 1.0²*vf1
    t_sqmean = 0.0*(vf0**2) + (0.5**2)*vf05 + (1.0**2)*vf1
    
    # Compute losses
    loss_m1 = F.mse_loss(m1, torch.full_like(m1, t_mean)) if w_m1 > 0 else torch.tensor(0., device=device)
    loss_m2 = F.mse_loss(m2, torch.full_like(m2, t_sqmean)) if w_m2 > 0 else torch.tensor(0., device=device)
    
    return loss_m1, loss_m2, m1.item()

