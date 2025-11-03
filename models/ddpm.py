"""
DDPM (Denoising Diffusion Probabilistic Models) scheduler
"""
import torch


class DDPM:
    """
    DDPM scheduler for diffusion models
    Implements forward diffusion and reverse sampling
    """
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2, device='cpu'):
        self.device = device
        self.num_timesteps = timesteps
        
        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Precompute useful quantities
        self.sqrt_acp = torch.sqrt(self.alphas_cumprod)
        self.sqrt_om_acp = torch.sqrt(1 - self.alphas_cumprod)
        
        # Posterior variance for sampling
        prev = torch.cat([torch.tensor([1.], device=device), self.alphas_cumprod[:-1]], dim=0)
        self.posterior_variance = betas * (1 - prev) / (1 - self.alphas_cumprod)

    def p_sample(self, model, x_t, t):
        """
        Single reverse diffusion step
        
        Args:
            model: Noise prediction model
            x_t: Noisy latent at timestep t
            t: Current timestep
        Returns:
            x_{t-1}: Less noisy latent
        """
        b = t.shape[0]
        coef1 = 1 / torch.sqrt(self.alphas[t]).view(b, 1, 1, 1)
        coef2 = self.betas[t].view(b, 1, 1, 1) / self.sqrt_om_acp[t].view(b, 1, 1, 1)
        
        # Predict noise
        pred = model(x_t, t)
        
        # Compute mean
        mean = coef1 * (x_t - coef2 * pred)
        
        # Add noise (except at t=0)
        noise = torch.randn_like(x_t) if (t > 0).any() else torch.zeros_like(x_t)
        var = self.posterior_variance[t].view(b, 1, 1, 1)
        
        return mean + torch.sqrt(var) * noise

