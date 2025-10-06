from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

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
