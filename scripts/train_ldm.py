import os
import argparse
import math
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm

# ------------------------------
# Your CustomVAE definition
# ------------------------------
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
        x = self.down(x)
        return self.act(self.gn(x))

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
        x = self.up(x)
        return self.act(self.gn(x))

class CustomVAE(nn.Module):
    def __init__(self, latent_ch=4):
        super().__init__()
        # encoder: 64→32→16
        self.conv_in = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.down1  = DownBlock(128, 128)
        self.down2  = DownBlock(128, 256)
        self.res1   = ResidualBlock(256)
        self.attn1  = AttentionBlock(256)
        self.res2   = ResidualBlock(256)
        self.to_mu     = nn.Conv2d(256, latent_ch, kernel_size=1)
        self.to_logvar = nn.Conv2d(256, latent_ch, kernel_size=1)

        # decoder: 16→32→64
        self.conv_z = nn.Conv2d(latent_ch, 256, kernel_size=3, padding=1)
        self.res3   = ResidualBlock(256)
        self.attn2  = AttentionBlock(256)
        self.res4   = ResidualBlock(256)
        self.up1    = UpBlock(256, 128)
        self.up2    = UpBlock(128, 64)
        self.conv_out = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.act_out  = nn.Sigmoid()

    def encode(self, x):  # x∈[-1,1]
        h = self.conv_in(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.res1(h)
        h = self.attn1(h)
        h = self.res2(h)
        mu     = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

    def decode(self, z):  # returns ∈[0,1]
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
# DDPM scheduler (unchanged)
# ----------------------------------------
class DDPM:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2, device='cpu'):
        self.num_timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.], device=device),
            self.alphas_cumprod[:-1]
        ], dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1 - self.alphas_cumprod_prev) /
            (1 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        noise = noise if noise is not None else torch.randn_like(x_start)
        a = self.sqrt_alphas_cumprod[t].view(-1,1,1,1)
        bm = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
        return a * x_start + bm * noise

    def p_sample(self, model, x_t, t):
        betas_t = self.betas[t].view(-1,1,1,1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
        pred_noise = model(x_t, t)
        coef1 = 1/torch.sqrt(self.alphas[t]).view(-1,1,1,1)
        coef2 = betas_t / sqrt_om
        mean = coef1 * (x_t - coef2 * pred_noise)
        var  = self.posterior_variance[t].view(-1,1,1,1)
        noise = torch.randn_like(x_t) if (t>0).any() else torch.zeros_like(x_t)
        return mean + torch.sqrt(var) * noise

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

# ----------------------------------------
# Training loop with intermediate saves and final latent dumps
# ----------------------------------------
def train_ldm_vae(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    tfm = transforms.Compose([
        transforms.Grayscale(1), transforms.Resize((64,64)), transforms.ToTensor()
    ])
    loader = DataLoader(
        datasets.ImageFolder(args.data_dir, transform=tfm),
        batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    ckpt = torch.load(args.vae_ckpt, map_location=device)
    vae = CustomVAE(latent_ch=args.latent_ch).to(device)
    vae.load_state_dict(ckpt['vae'])
    vae.eval(); [p.requires_grad_(False) for p in vae.parameters()]

    unet = TimeUNet(args.latent_ch).to(device)
    ddpm = DDPM(timesteps=args.num_train_timesteps, beta_start=args.beta_start,
                beta_end=args.beta_end, device=device)
    opt = torch.optim.Adam(unet.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    os.makedirs(args.save_dir, exist_ok=True)
    log = open(os.path.join(args.save_dir,'train.log'),'w')

    H=W=16
    for ep in range(1,args.epochs+1):
        unet.train(); tot=0
        for imgs,_ in tqdm(loader, desc=f"Epoch {ep}/{args.epochs}"):
            imgs=imgs.to(device); x_norm=2*imgs-1
            with torch.no_grad(): mu,lv=vae.encode(x_norm); lat=reparameterize(mu,lv)
            noise=torch.randn_like(lat)
            t_idx=torch.randint(0, ddpm.num_timesteps,(lat.size(0),),device=device)
            x_t=ddpm.q_sample(lat,t_idx,noise)
            pred=unet(x_t,t_idx); loss=mse(pred,noise)
            opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
        avg=tot/len(loader); log.write(f"{ep},{avg:.4f}\n"); print(f"Epoch {ep} Loss={avg:.4f}")

        # sampling with intermediate images
        if ep%args.save_interval==0 or ep==args.epochs:
            with torch.no_grad():
                torch.save(unet.state_dict(), os.path.join(args.save_dir,f'unet_ep{ep}.pth'))
                unet.eval()
                xq = torch.randn(args.num_samples,args.latent_ch,H,W,device=device)
                os.makedirs(os.path.join(args.save_dir,f"samps{ep}"),exist_ok=True)
                for ts in range(ddpm.num_timesteps-1,-1,-1):
                    t_q=torch.full((args.num_samples,),ts,dtype=torch.long,device=device)
                    xq = ddpm.p_sample(unet,xq,t_q)
                    # decode and save intermediate
                    imgs_out = vae.decode(xq).clamp(0,1)
                    save_image(imgs_out, os.path.join(args.save_dir,f"samps{ep}",f"s{ts:04d}.png"), normalize=False)
                # after final decode, also dump latent channels
                for c in range(args.latent_ch):
                    zc = xq[:,c:c+1,:,:]
                    save_image(zc, os.path.join(args.save_dir,f'latent_c{c}_ep{ep}.png'), normalize=True)
                unet.train()
    log.close()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data_dir',required=True)
    p.add_argument('--vae_ckpt',required=True)
    p.add_argument('--save_dir',default='./ldm_custom')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch_size',type=int,default=16)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--latent_ch',type=int,default=4)
    p.add_argument('--num_train_timesteps',type=int,default=1000)
    p.add_argument('--beta_start',type=float,default=1e-4)
    p.add_argument('--beta_end',type=float,default=2e-2)
    p.add_argument('--save_interval',type=int,default=10)
    p.add_argument('--num_samples',type=int,default=16)
    args=p.parse_args()
    train_ldm_vae(args)
