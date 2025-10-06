import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TorchFEMMesh(nn.Module):
    """
    Differentiable FEM solver for a 2D MxN grid with Dirichlet BC on left (u=1) and right (u=0)
    Computes effective diffusivity D_eff for a continuous mask.
    """
    def __init__(self, M:int, N:int, low_cond:float=0.001, device='cpu'):
        super().__init__()
        self.M, self.N = M, N
        self.nx = N + 1
        self.ny = M + 1
        self.nn = self.nx * self.ny
        self.low_cond = low_cond
        self.device = device

        # reference gradients for linear triangle
        Dhat = np.array([[-1,-1],[1,0],[0,1]], dtype=float)

        # build element connectivity (two triangles per cell)
        elems = []
        for j in range(M):
            for i in range(N):
                n0 = j*self.nx + i
                n1 = n0 + 1
                n2 = n0 + self.nx
                n3 = n2 + 1
                elems.append((n0,n1,n2))
                elems.append((n1,n3,n2))
        elems = np.array(elems, dtype=np.int64)  # [n_elems,3]
        self.register_buffer('elems', torch.from_numpy(elems))
        n_elems = elems.shape[0]

        # assemble base stiffness entries
        rows, cols, base_data = [], [], []
        elem_idx = []
        gradT_list = []
        area_list = []
        for idx, e in enumerate(elems):
            # node coordinates
            pts = np.array([[e[k] % self.nx, e[k]//self.nx] for k in range(3)], float)
            J = np.vstack((pts[1]-pts[0], pts[2]-pts[0])).T
            detJ = np.linalg.det(J)
            invJT = np.linalg.inv(J).T
            grads = Dhat.dot(invJT)           # [3,2]
            gradT = grads.T                   # [2,3]
            area = abs(detJ) / 2.0
            ke0 = area * (grads.dot(grads.T))  # [3,3]
            # store stiffness pattern
            for a in range(3):
                for b in range(3):
                    rows.append(e[a])
                    cols.append(e[b])
                    base_data.append(ke0[a,b])
                    elem_idx.append(idx)
            gradT_list.append(gradT)
            area_list.append(area)

        # register sparse assembly buffers
        self.register_buffer('rows',     torch.LongTensor(rows))       # [nnz]
        self.register_buffer('cols',     torch.LongTensor(cols))       # [nnz]
        self.register_buffer('base_data',torch.tensor(base_data, dtype=torch.float32))
        self.register_buffer('elem_idx', torch.LongTensor(elem_idx))   # [nnz]
        # element grads and areas
        self.register_buffer('elem_gradT', torch.tensor(np.stack(gradT_list), dtype=torch.float32))  # [n_elems,2,3]
        self.register_buffer('elem_area',  torch.tensor(area_list,    dtype=torch.float32))          # [n_elems]

        # Dirichlet BC: left boundary =1, right boundary=0
        bc = []
        for j in range(self.ny):
            bc.append(j*self.nx + 0)
            bc.append(j*self.nx + (self.nx-1))
        bc_idx = np.unique(bc)
        fc_idx = np.setdiff1d(np.arange(self.nn), bc_idx)
        self.register_buffer('bc_idx', torch.LongTensor(bc_idx))
        self.register_buffer('fc_idx', torch.LongTensor(fc_idx))
        # fixed values u_c
        u_c = np.zeros(len(bc_idx), dtype=np.float32)
        # left boundary =1, right=0
        coords = np.stack([bc_idx % self.nx, bc_idx // self.nx],-1)
        u_c[coords[:,0]==0] = 1.0
        u_c[coords[:,0]==self.nx-1] = 0.0
        self.register_buffer('u_c', torch.tensor(u_c))

        # total area for normalization D_eff
        self.register_buffer('total_area', torch.tensor(float(self.M*self.N)))

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        mask: [H,W] float in [0,1] gives conductivity map per cell
        returns: scalar D_eff
        """
        # element conductivities
        m_flat = mask.reshape(-1)                      # [M*N]
        sigma_e = self.low_cond + (1.0-self.low_cond) * m_flat.repeat_interleave(2)  # [n_elems]
        # assemble global stiffness
        data = self.base_data * sigma_e[self.elem_idx]
        K = torch.sparse_coo_tensor(
            torch.stack([self.rows, self.cols], 0),
            data,
            (self.nn, self.nn),
            device=self.device
        )
        # turn to dense so we can index with fancy indexing
        K_dense = K.to_dense()           # [nn,nn]
        bc, fc = self.bc_idx, self.fc_idx
        K_ff = K_dense[fc][:, fc]        # [n_free, n_free]
        K_fc = K_dense[fc][:, bc]        # [n_free, n_bc]
        b_f  = - K_fc @ self.u_c         # [n_free]
        # solve the linear system directly — autograd-friendly
        u_f  = torch.linalg.solve(K_ff, b_f)




        # build full solution
        u = torch.zeros(self.nn, device=self.device)
        u[fc] = u_f
        u[bc] = self.u_c
        # compute D_eff by energy
        # gather per-element u_e
        u_e = u[self.elems]                               # [n_elems,3]
        grad_u = torch.matmul(self.elem_gradT, u_e.unsqueeze(-1)).squeeze(-1)  # [n_elems,2]
        sq = (grad_u**2).sum(dim=1)                       # [n_elems]
        Deff = (sigma_e * sq * self.elem_area).sum() # / self.total_area
        return Deff



def _make_gaussian_kernel(kernel_size:int, sigma:float, device):
    """Returns a [1,1,ks,ks] Gaussian kernel for conv2d."""
    ax = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2
    xx, yy = torch.meshgrid(ax, ax)
    kernel = torch.exp(-(xx**2 + yy**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1,1,kernel_size,kernel_size)


def radial_profile(corr):
    h, w = corr.shape
    cy, cx = h//2, w//2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    max_r = r.max()
    sum_r = np.bincount(r.ravel(), weights=corr.ravel(), minlength=max_r+1)
    count_r = np.bincount(r.ravel(), minlength=max_r+1)
    return sum_r / np.maximum(count_r, 1)

def compute_autocorrelation_2d(mask):
    """
    Periodic‐BC autocorrelation of a single phase mask (2D).
    Returns radial_profile of S(r).
    """
    h, w = mask.shape
    n = h*w
    F = np.fft.fft2(mask)
    corr = np.fft.ifft2(F * np.conj(F))
    corr = np.real(corr)/n
    corr = np.fft.fftshift(corr)
    return radial_profile(corr)

def average_slice_autocorr(volume_labels, phases):
    """
    volume_labels: integer array [Z,Y,X] with values in phases.
    Returns dict { axis_name: { p: mean_profile } }.
    """
    Z, Y, X = volume_labels.shape
    out = {'z':{}, 'y':{}, 'x':{}}
    # for each axis, collect slice‐wise autocorr and average
    for axis, name, L in [(0,'z',Z), (1,'y',Y), (2,'x',X)]:
        for p in phases:
            profs = []
            for i in range(L):
                if axis==0: sl = volume_labels[i,:,:]
                if axis==1: sl = volume_labels[:,i,:]
                if axis==2: sl = volume_labels[:,:,i]
                mask = (sl==p).astype(np.float32)
                profs.append(compute_autocorrelation_2d(mask))
            out[name][p] = np.mean(np.stack(profs,axis=0), axis=0)
    return out

# ----------------------------------------
# CustomVAE definition (from training script)
# ----------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels, num_groups=16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn1   = nn.GroupNorm(num_groups, channels)
        self.act1  = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn2   = nn.GroupNorm(num_groups, channels)
        self.act2  = nn.SiLU(inplace=True)

    def forward(self, x):
        h = self.act1(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return self.act2(x + h)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(16, channels)
        self.qkv  = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        q, k, v = self.qkv(x_norm).chunk(3, dim=1)
        q = q.view(b, c, -1).permute(0, 2, 1)
        k = k.view(b, c, -1)
        v = v.view(b, c, -1).permute(0, 2, 1)
        attn = torch.bmm(q, k) / math.sqrt(c)
        attn = torch.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)
        out = out.permute(0, 2, 1).view(b, c, h, w)
        return x + self.proj_out(out)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res1 = ResidualBlock(in_ch)
        self.res2 = ResidualBlock(in_ch)
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.gn   = nn.GroupNorm(16, out_ch)
        self.act  = nn.SiLU(inplace=True)
    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        return self.act(self.gn(self.down(x)))

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res1 = ResidualBlock(in_ch)
        self.res2 = ResidualBlock(in_ch)
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.gn   = nn.GroupNorm(16, out_ch)
        self.act  = nn.SiLU(inplace=True)
    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        return self.act(self.gn(self.up(x)))

class CustomVAE(nn.Module):
    def __init__(self, latent_ch=4):
        super().__init__()
        # encoder: 64→32→16
        self.conv_in = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.down1   = DownBlock(128, 128)
        self.down2   = DownBlock(128, 256)
        self.res1    = ResidualBlock(256)
        self.attn1   = AttentionBlock(256)
        self.res2    = ResidualBlock(256)
        self.to_mu     = nn.Conv2d(256, latent_ch, kernel_size=1)
        self.to_logvar = nn.Conv2d(256, latent_ch, kernel_size=1)
        # decoder: 16→32→64
        self.conv_z  = nn.Conv2d(latent_ch, 256, kernel_size=3, padding=1)
        self.res3    = ResidualBlock(256)
        self.attn2   = AttentionBlock(256)
        self.res4    = ResidualBlock(256)
        self.up1     = UpBlock(256, 128)
        self.up2     = UpBlock(128, 64)
        self.conv_out = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.act_out  = nn.Sigmoid()

    def encode(self, x):  # x ∈ [-1,1]
        h = self.conv_in(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.res1(h)
        h = self.attn1(h)
        h = self.res2(h)
        mu     = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

    def decode(self, z):  # returns ∈ [0,1]
        h = self.conv_z(z)
        h = self.res3(h)
        h = self.attn2(h)
        h = self.res4(h)
        h = self.up1(h)
        h = self.up2(h)
        return self.act_out(self.conv_out(h))

def reparameterize(mu, logvar):
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std

# ----------------------------------------
# DDPM scheduler
# ----------------------------------------
class DDPM:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2, device='cpu'):
        self.device = device
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_acp = torch.sqrt(self.alphas_cumprod)
        self.sqrt_om_acp = torch.sqrt(1 - self.alphas_cumprod)
        prev = torch.cat([torch.tensor([1.], device=device), self.alphas_cumprod[:-1]], dim=0)
        self.posterior_variance = betas * (1 - prev) / (1 - self.alphas_cumprod)
        self.num_timesteps = timesteps

    def p_sample(self, model, x_t, t):
        b = t.shape[0]
        coef1 = 1 / torch.sqrt(self.alphas[t]).view(b, 1, 1, 1)
        coef2 = self.betas[t].view(b, 1, 1, 1) / self.sqrt_om_acp[t].view(b, 1, 1, 1)
        pred = model(x_t, t)
        mean = coef1 * (x_t - coef2 * pred)
        noise = torch.randn_like(x_t) if (t > 0).any() else torch.zeros_like(x_t)
        var = self.posterior_variance[t].view(b, 1, 1, 1)
        return mean + torch.sqrt(var) * noise

# ----------------------------------------
# TimeUNet for SDS
# ----------------------------------------
# ----------------------------------------
# Enhanced Time UNet with widened channels, multi-scale attention, and deeper time MLP
# ----------------------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin1 = nn.Linear(dim, dim * 4)
        self.act1 = nn.SiLU()
        self.lin2 = nn.Linear(dim * 4, dim * 4)
        self.act2 = nn.SiLU()

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args  = t.float()[:, None] * freqs[None]
        emb   = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        h = self.act1(self.lin1(emb))
        h = self.act2(self.lin2(h))
        return h

class ResidualBlockUNet(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.skip     = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm1    = nn.GroupNorm(min(16, in_ch), in_ch)
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2    = nn.GroupNorm(min(16, out_ch), out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim * 4, out_ch)

    def forward(self, x, te):
        h = F.silu(self.conv1(self.norm1(x))) + self.time_mlp(te)[:, :, None, None]
        h = F.silu(self.conv2(self.norm2(h)))
        return h + self.skip(x)

class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(min(16, ch), ch)
        self.qkv  = nn.Conv1d(ch, ch * 3, 1)
        self.proj = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        flat = self.norm(x).view(b, c, -1)
        q,k,v = self.qkv(flat).chunk(3, dim=1)
        q,k,v = [t.permute(0,2,1) for t in (q,k,v)]
        attn = torch.softmax(q @ k.transpose(-2,-1) / math.sqrt(c), dim=-1)
        o    = (attn @ v).permute(0,2,1)
        return x + self.proj(o).view(b, c, h, w)

class TimeUNet(nn.Module):
    def __init__(self, latent_ch, base_ch=128, time_dim=64):
        super().__init__()
        self.time_emb   = TimeEmbedding(time_dim)
        self.enc1       = ResidualBlockUNet(latent_ch, base_ch, time_dim)
        self.attn16     = SelfAttention(base_ch)
        self.pool1      = nn.MaxPool2d(2)
        self.enc2       = ResidualBlockUNet(base_ch, base_ch*2, time_dim)
        self.attn8      = SelfAttention(base_ch*2)
        self.pool2      = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlockUNet(base_ch*2, base_ch*4, time_dim)
        self.attn4      = SelfAttention(base_ch*4)
        self.up2        = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2,2)
        self.dec2       = ResidualBlockUNet(base_ch*4, base_ch*2, time_dim)
        self.up1        = nn.ConvTranspose2d(base_ch*2, base_ch, 2,2)
        self.dec1       = ResidualBlockUNet(base_ch*2, base_ch, time_dim)
        self.out        = nn.Conv2d(base_ch, latent_ch, 1)

    def forward(self, x, t):
        te = self.time_emb(t)
        e1 = self.enc1(x, te)
        e1 = self.attn16(e1)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1, te)
        e2 = self.attn8(e2)
        p2 = self.pool2(e2)
        b  = self.bottleneck(p2, te)
        b  = self.attn4(b)
        d2 = self.dec2(torch.cat([self.up2(b), e2], 1), te)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1), te)
        return self.out(d1)


def average_slice_sa(volume, phases, sigma=1.0):
    """
    volume: 3D int array [Z,Y,X] of labels in `phases`
    phases: list of ints
    sigma: Gaussian blur std-dev (pixels)

    Returns: dict { 'z':{p:rel_sa}, 'y':{…}, 'x':{…} }
    where rel_sa is the mean over slices of:
        SA_p(slice) / sum_q SA_q(slice)
    """
    from scipy.ndimage import gaussian_filter

    Z, Y, X = volume.shape
    out = {'z':{}, 'y':{}, 'x':{}}

    # for each axis, we'll accumulate per-slice relative SA, then average:
    for axis, name, L in [(0,'z',Z), (1,'y',Y), (2,'x',X)]:
        # initialize sum of relative SAs
        rel_sums = {p: 0.0 for p in phases}

        for i in range(L):
            # extract slice
            if axis == 0:
                sl = volume[i,:,:]
            elif axis == 1:
                sl = volume[:,i,:]
            else:
                sl = volume[:,:,i]

            # compute absolute SA for each phase on this slice
            sa_slice = {}
            for p in phases:
                m = (sl == p).astype(np.float32)
                m_s = gaussian_filter(m, sigma=sigma)
                dx = np.abs(m_s[1: , :] - m_s[:-1, :])
                dy = np.abs(m_s[ : , 1:] - m_s[:, :-1])
                sa_slice[p] = float((dx.sum() + dy.sum())/(X*Y))

            # accumulate each phase's relative SA for this slice
            for p in phases:
                rel_sums[p] += sa_slice[p]

        # average over slices
        for p in phases:
            out[name][p] = rel_sums[p] / L

    return out


def compute_relative_SA_torch(
    recon: torch.Tensor,
    phases: list[int],
    kernel_size: int = 7,
    sigma: float = 1.0,
    beta: float = 50.0,
    eps: float = 1e-8
) -> torch.Tensor:
    B,_,H,W = recon.shape
    device, dtype = recon.device, recon.dtype

    # derive P from the passed phases list
    P = len(phases)

    # 1) ideal discrete levels
    levels = torch.linspace(0.0, 1.0, steps=P, device=device, dtype=dtype)  # [P]
    # broadcast ...
    x = recon.expand(B, P, H, W)                                # [B,P,H,W]
    l = levels.view(1, P, 1, 1)                                  # [1,P,1,1]

    # 2) soft‐assignment mask
    dist  = torch.abs(x - l)                                     # [B,P,H,W]
    masks = F.softmax(-beta * dist.view(B, P, -1), dim=1)        # [B,P,H*W]
    masks = masks.view(B, P, H, W)                               # [B,P,H,W]

    # 3) Gaussian smoothing
    gk    = _make_gaussian_kernel(kernel_size, sigma, device)    # [1,1,ks,ks]
    gk    = gk.repeat(P, 1, 1, 1)                                # [P,1,ks,ks]
    M_s   = F.conv2d(masks, weight=gk,
                     padding=kernel_size//2, groups=P)         # [B,P,H,W]

    # 4) total variation
    tv_h = torch.abs(M_s[:,:,1:,:] - M_s[:,:,:-1,:]).sum(dim=(2,3))  # [B,P]
    tv_w = torch.abs(M_s[:,:,:,1:] - M_s[:,:,:,:-1]).sum(dim=(2,3))  # [B,P]
    sa   = (tv_h + tv_w) / (H*W)                                    # [B,P]

    # 5) relative SA
    rel = sa  # [B,P]
    return rel.mean(dim=0)                          # [P]



