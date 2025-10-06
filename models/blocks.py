import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, ch, groups=16):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1); self.n1 = nn.GroupNorm(groups, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1); self.n2 = nn.GroupNorm(groups, ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        h = self.act(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return self.act(x + h)

class AttentionBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm  = nn.GroupNorm(16, ch)
        self.to_qkv = nn.Conv2d(ch, ch*3, 1)
        self.proj   = nn.Conv2d(ch, ch, 1)
    def forward(self, x):
        b,c,h,w = x.shape
        q,k,v = self.to_qkv(self.norm(x)).chunk(3,1)
        q = q.reshape(b,c,h*w).permute(0,2,1)
        k = k.reshape(b,c,h*w)
        v = v.reshape(b,c,h*w).permute(0,2,1)
        attn = torch.softmax(torch.bmm(q,k)/math.sqrt(c), dim=-1)
        out  = torch.bmm(attn,v).permute(0,2,1).reshape(b,c,h,w)
        return x + self.proj(out)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res = nn.Sequential(ResidualBlock(in_ch), ResidualBlock(in_ch))
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1),
            nn.GroupNorm(16, out_ch),
            nn.SiLU(inplace=True)
        )
    def forward(self, x): return self.down(self.res(x))

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res = nn.Sequential(ResidualBlock(in_ch), ResidualBlock(in_ch))
        self.up  = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.GroupNorm(16, out_ch),
            nn.SiLU(inplace=True)
        )
    def forward(self, x): return self.up(self.res(x))

# ───────────────────────── Custom VAE ─────────────────────────
