"""
Effective diffusivity computation and matching via FEM
"""
import torch
import torch.nn.functional as F


def compute_diffusivity_loss(masks, fem_solver, rd_targets, phases, device):
    """
    Compute effective diffusivity matching loss using FEM
    
    Args:
        masks: [1, P, H, W] soft phase probability masks
        fem_solver: TorchFEMMesh instance
        rd_targets: Dict mapping phase -> target diffusivity
        phases: List of phase indices
        device: torch device
    Returns:
        loss_rd: Scalar diffusivity matching loss
    """
    P = len(phases)
    
    # Compute diffusivity for each phase
    deff = []
    for p in range(P):
        mask_p = masks[0, p]  # [H, W]
        deff.append(fem_solver(mask_p))
    
    deff = torch.stack(deff)  # [P]
    
    # Target diffusivities
    deff_targets = torch.tensor([rd_targets[ph] for ph in phases], device=device)
    
    # MSE loss
    return F.mse_loss(deff, deff_targets)

