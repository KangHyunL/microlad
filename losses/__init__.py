"""
Loss functions for physics-informed microstructure generation
"""
from .tpc import compute_tpc_torch, compute_tpc_loss_ste, setup_tpc_bins
from .volume_fraction import compute_vf_loss
from .surface_area import compute_sa_loss
from .diffusivity import compute_diffusivity_loss

__all__ = [
    'compute_tpc_torch',
    'compute_tpc_loss_ste',
    'setup_tpc_bins',
    'compute_vf_loss',
    'compute_sa_loss',
    'compute_diffusivity_loss'
]

