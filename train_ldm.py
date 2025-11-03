"""
Train Latent Diffusion Model (LDM)
"""
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from models import CustomVAE, TimeUNet


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


def train_epoch(unet, vae, loader, optimizer, device, timesteps=1000):
    unet.train()
    
    betas = torch.linspace(1e-4, 2e-2, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_acp = torch.sqrt(alphas_cumprod)
    sqrt_omacp = torch.sqrt(1.0 - alphas_cumprod)
    
    total_loss = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        x = batch.to(device)
        
        with torch.no_grad():
            mu, _ = vae.encode(x * 2 - 1)
            z = mu
        
        t = torch.randint(0, timesteps, (z.shape[0],), device=device)
        noise = torch.randn_like(z)
        
        z_t = sqrt_acp[t].view(-1, 1, 1, 1) * z + sqrt_omacp[t].view(-1, 1, 1, 1) * noise
        pred = unet(z_t, t)
        
        loss = F.mse_loss(pred, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description='Train LDM')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--vae_ckpt', required=True)
    parser.add_argument('--save_dir', default='./ldm_checkpoints')
    parser.add_argument('--latent_ch', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--base_ch', type=int, default=128)
    parser.add_argument('--time_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_interval', type=int, default=20)
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])
    
    dataset = MicrostructureDataset(args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    print("Loading VAE...")
    vae = CustomVAE(latent_ch=args.latent_ch).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device)['vae'])
    vae.eval()
    [p.requires_grad_(False) for p in vae.parameters()]
    
    unet = TimeUNet(args.latent_ch, base_ch=args.base_ch, time_dim=args.time_dim).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)
    
    print(f"\nTraining LDM: {len(dataset)} images, {args.epochs} epochs\n")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        avg_loss = train_epoch(unet, vae, loader, optimizer, device, timesteps=args.timesteps)
        print(f"  Loss: {avg_loss:.6f}")
        
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            torch.save(unet.state_dict(), os.path.join(args.save_dir, f'unet_ep{epoch}.pth'))
            print(f"  Saved checkpoint")
    
    print(f"\nComplete! Checkpoints: {args.save_dir}")


if __name__ == '__main__':
    main()

