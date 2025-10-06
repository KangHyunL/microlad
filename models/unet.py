import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..models.blocks import ResidualBlock, AttentionBlock

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

# ----------------------------------------
# Training loop with intermediate saves and final latent dumps
# ----------------------------------------
