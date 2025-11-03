"""
Train VAE for microstructure images
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from pytorch_msssim import ssim

from models import CustomVAE, reparameterize


class MicrostructureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.paths = []
        for ext in ["png", "jpg", "bmp", "tif", "tiff"]:
            self.paths += list(sorted(Path(root_dir).rglob(f"*.{ext}")))
        self.transform = transform
        print(f"Found {len(self.paths)} images")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        x = self.transform(img) if self.transform else transforms.ToTensor()(img)
        return x


def vae_loss(recon, x, mu, logvar, beta=1.0):
    """VAE loss with reconstruction and KL divergence"""
    mse_loss = F.mse_loss(recon, x)
    ssim_loss = 1 - ssim(recon, x, data_range=1.0, size_average=True)
    recon_loss = mse_loss + 0.1 * ssim_loss
    
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / x.numel()
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_epoch(vae, loader, optimizer, device, beta):
    vae.train()
    total, recon_sum, kl_sum = 0.0, 0.0, 0.0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        x = batch.to(device)
        mu, logvar = vae.encode(x * 2 - 1)
        z = reparameterize(mu, logvar)
        recon = vae.decode(z)
        
        loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total += loss.item()
        recon_sum += recon_loss.item()
        kl_sum += kl_loss.item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'recon': f'{recon_loss.item():.4f}', 'kl': f'{kl_loss.item():.6f}'})
    
    n = len(loader)
    return total/n, recon_sum/n, kl_sum/n


def main():
    parser = argparse.ArgumentParser(description='Train VAE')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--save_dir', default='./vae_checkpoints')
    parser.add_argument('--latent_ch', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_interval', type=int, default=10)
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])
    
    dataset = MicrostructureDataset(args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    vae = CustomVAE(latent_ch=args.latent_ch).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)
    
    print(f"\nTraining VAE: {len(dataset)} images, {args.epochs} epochs\n")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        avg_loss, avg_recon, avg_kl = train_epoch(vae, loader, optimizer, device, args.beta)
        print(f"  Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.6f}")
        
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            torch.save({'vae': vae.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': vars(args)},
                      os.path.join(args.save_dir, f'vae_epoch{epoch}.pth'))
            
            vae.eval()
            with torch.no_grad():
                sample = next(iter(loader))[:8].to(device)
                mu, logvar = vae.encode(sample * 2 - 1)
                recon = vae.decode(reparameterize(mu, logvar))
                save_image(torch.cat([sample, recon]), os.path.join(args.save_dir, f'recon_epoch{epoch}.png'), nrow=8)
            vae.train()
    
    print(f"\nComplete! Checkpoints: {args.save_dir}")


if __name__ == '__main__':
    main()

