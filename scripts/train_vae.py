#!/usr/bin/env python
# vae_latent_pretrain.py  (with per-epoch evaluation)
# -------------------------------------------------------------
# VAE comparison (latent 4×4 / 8×8 / 16×16) + optional pre-training
# Metrics: MAE, PSNR, SSIM, Latent Consistency
# -------------------------------------------------------------
import os, math, csv, argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from pytorch_msssim import ssim
from PIL import Image
from tqdm.auto import tqdm

# ───────────────────────── Dataset ────────────────────────────
class MicrostructureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.paths = []
        for ext in ("png", "jpg", "bmp", "tif"):
            self.paths += list(sorted(Path(root_dir).rglob(f"*.{ext}")))
        if not self.paths:
            raise RuntimeError(f"No images found in {root_dir}")
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        return self.transform(img)  # tensor in [0,1]

# ───────────────────────── Building blocks ────────────────────
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
class CustomVAE(nn.Module):
    def __init__(self, latent_ch=4, latent_grid=16):
        super().__init__()
        assert latent_grid in {4,8,16}
        self.latent_grid    = latent_grid
        self.scaling_factor = 1.0

        # encoder
        self.enc = nn.ModuleList([
            nn.Conv2d(1, 128, 3, 1, 1),
            DownBlock(128, 128),  # 64→32
            DownBlock(128, 256),  # 32→16
        ])
        self.mid        = nn.Sequential(ResidualBlock(256), AttentionBlock(256), ResidualBlock(256))
        self.to_mu      = nn.Conv2d(256, latent_ch, 1)
        self.to_logvar  = nn.Conv2d(256, latent_ch, 1)

        # decoder
        self.conv_z   = nn.Conv2d(latent_ch, 256, 3, 1, 1)
        self.mid_dec  = nn.Sequential(ResidualBlock(256), AttentionBlock(256), ResidualBlock(256))
        self.up1      = UpBlock(256, 128)  # 16→32
        self.up2      = UpBlock(128,  64)  # 32→64
        self.conv_out = nn.Conv2d(64, 1, 3, 1, 1)
        self.act_out  = nn.Sigmoid()

    def _pool(self, t, mode="down"):
        if self.latent_grid == 16:
            return t
        factor = 16 // self.latent_grid
        if mode == "down":
            return F.avg_pool2d(t, kernel_size=factor)
        else:
            return F.interpolate(t, size=(16,16), mode="nearest")

    def encode(self, x):
        for m in self.enc:
            x = m(x)
        h = self.mid(x)
        mu, logvar = self.to_mu(h), self.to_logvar(h)
        return self._pool(mu, "down"), self._pool(logvar, "down")

    def decode(self, z):
        z = self._pool(z, "up")
        h = self.conv_z(z)
        h = self.mid_dec(h)
        h = self.up1(h)
        h = self.up2(h)
        return self.act_out(self.conv_out(h))

# ───────────────────────── Helpers ────────────────────────────
def reparam(mu, logvar):
    std = (0.5 * logvar).exp()
    return mu + torch.randn_like(std) * std

def psnr_tensor(x, y):
    mse = F.mse_loss(x, y, reduction="mean")
    return 10 * torch.log10(1.0 / mse)

def latent_consistency(model, img):
    # single-image consistency: horizontal vs vertical flip
    aug1 = TF.hflip(img)
    aug2 = TF.vflip(img)
    mu1, lv1 = model.encode(2*aug1.unsqueeze(0) - 1)
    mu2, lv2 = model.encode(2*aug2.unsqueeze(0) - 1)
    return torch.norm(mu1 - mu2, p=2).item()

@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    total_n = mae_sum = psnr_sum = ssim_sum = 0
    for x in loader:
        x = x.to(dev)
        mu, lv = model.encode(2*x - 1)
        rec    = model.decode(reparam(mu, lv))

        n = x.size(0)
        mae_sum  += F.l1_loss(rec, x, reduction="mean").item() * n
        psnr_sum += psnr_tensor(rec, x).item() * n
        ssim_sum += ssim(rec, x, data_range=1.0, size_average=False).mean().item() * n
        total_n += n

    return mae_sum/total_n, psnr_sum/total_n, ssim_sum/total_n

# ───────────────────────── Train stage ────────────────────────
def train_stage(model, loader, optimizer, epochs, beta, dev, tag, save_dir, interval):
    log_file = save_dir / "train_log.txt"
    with open(log_file, "a") as f:
        f.write(f"\n# {tag} started {datetime.now()}\nEpoch,Recon,KL\n")

    for ep in range(1, epochs+1):
        model.train()
        recon_accum = kl_accum = n = 0
        beta_eff = beta * min(1.0, ep/10)
        pbar = tqdm(loader, desc=f"[{tag}] Epoch {ep}/{epochs}", leave=False)

        for x in pbar:
            x = x.to(dev)
            mu, lv = model.encode(2*x - 1)
            rec     = model.decode(reparam(mu, lv))

            l1       = F.l1_loss(rec, x)
            ssim_l   = 1 - ssim(rec, x, data_range=1.0, size_average=True)
            recon_l  = l1 + 0.5 * ssim_l
            kl_loss  = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            loss     = recon_l + beta_eff * kl_loss

            optimizer.zero_grad(); loss.backward(); optimizer.step()

            b = x.size(0)
            recon_accum += recon_l.item() * b
            kl_accum    += kl_loss.item() * b
            n           += b
            pbar.set_postfix(R=f"{recon_l:.4f}", KL=f"{kl_loss:.4f}")

        # compute averages
        avg_recon = recon_accum / n
        avg_kl    = kl_accum / n

        # ───────── per-epoch evaluation ─────────
        model.eval()
        x_val = next(iter(loader))[:8].to(dev)
        with torch.no_grad():
            mu_val, lv_val = model.encode(2*x_val - 1)
            z_val          = reparam(mu_val, lv_val)
            recon_val      = model.decode(z_val)

            ssim_val = ssim(recon_val, x_val, data_range=1.0, size_average=True).item()
            latcons  = sum(latent_consistency(model, img) for img in x_val) / x_val.size(0)

        print(f"\nEpoch {ep} → Recon={avg_recon:.4f}, KL={avg_kl:.4f}, SSIM={ssim_val:.4f}, LatCons={latcons:.4f}\n")

        with open(log_file, "a") as f:
            f.write(f"{ep},{avg_recon:.4f},{avg_kl:.4f}\n")

        # save checkpoints & example reconstructions
        if ep % interval == 0 or ep == epochs:
            model.eval()
            sample = next(iter(loader))[:8].to(dev)
            recon_samp = model.decode(reparam(*model.encode(2*sample - 1)))
            save_image(sample,     save_dir / f"{tag}_orig_ep{ep}.png")
            save_image(recon_samp, save_dir / f"{tag}_rec_ep{ep}.png")
            torch.save({"vae": model.state_dict()}, save_dir / f"vae_{tag}_ep{ep}.pth")

# ───────────────────────── One run: pretrain+finetune ─────────
def run_once(args, grid, dev):
    save_dir = Path(args.save_root) / f"latent_{grid}x{grid}"
    save_dir.mkdir(parents=True, exist_ok=True)

    tf = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
    ds_main = MicrostructureDataset(args.data_dir, tf)
    dl_main = DataLoader(ds_main, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dl_eval = DataLoader(ds_main, batch_size=args.batch_size, shuffle=False, num_workers=4)

    dl_pre = None
    if args.data_dir_pretrain:
        ds_pre = MicrostructureDataset(args.data_dir_pretrain, tf)
        dl_pre = DataLoader(ds_pre, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = CustomVAE(latent_ch=args.latent_ch, latent_grid=grid).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if dl_pre:
        train_stage(model, dl_pre, optimizer, args.pretrain_epochs, args.beta,
                    dev, "pretrain", save_dir, args.save_interval)

    train_stage(model, dl_main, optimizer, args.epochs, args.beta,
                dev, "finetune", save_dir, args.save_interval)

    return (*evaluate(model, dl_eval, dev), str(save_dir))

# ─────────────────────────── main ─────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",         type=str, required=True)
    parser.add_argument("--data_dir_pretrain",type=str, default=None)
    parser.add_argument("--save_root",        type=str, default="./vae_latent_runs")
    parser.add_argument("--device_num",       type=str, default="0")
    parser.add_argument("--pretrain_epochs",  type=int, default=50)
    parser.add_argument("--epochs",           type=int, default=50)
    parser.add_argument("--batch_size",       type=int, default=32)
    parser.add_argument("--lr",               type=float, default=1e-4)
    parser.add_argument("--beta",             type=float, default=0.5)
    parser.add_argument("--latent_ch",        type=int, default=4)
    parser.add_argument("--save_interval",    type=int, default=10)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_num
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ensure save_root exists
    Path(args.save_root).mkdir(parents=True, exist_ok=True)

    summary = Path(args.save_root) / "metrics_summary.csv"
    write_header = not summary.exists()
    with open(summary, "a", newline="") as csvf:
        writer = csv.writer(csvf)
        if write_header:
            writer.writerow(["latent_grid", "MAE", "PSNR", "SSIM", "run_folder"])

        for g in (4, 8, 16):
            mae, psnr_v, ssim_v, folder = run_once(args, g, dev)
            writer.writerow([f"{g}x{g}", f"{mae:.6f}", f"{psnr_v:.2f}", f"{ssim_v:.4f}", folder])
            print(f"\n✓ Completed {g}×{g} → MAE={mae:.6f}, PSNR={psnr_v:.2f}, SSIM={ssim_v:.4f}\n")

if __name__ == "__main__":
    main()
