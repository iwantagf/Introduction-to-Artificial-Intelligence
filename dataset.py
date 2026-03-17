import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import os
import random


# Edge Extraction RCAS
def to_grayscale(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB to grayscale using standard luminance formula: 0.299R + 0.587G + 0.114B."""
    # rgb shape: (B, 3, H, W)
    return 0.299 * rgb[:, 0:1, :, :] + 0.587 * rgb[:, 1:2, :, :] + 0.114 * rgb[:, 2:3, :, :]


def fsr_edge(x):
    """
    Compute edge map using FSR-style luminance difference analysis.
    Ideally approximates the 'len' (edge strength) parameter in FSR 1.0 EASU.
    """
    if x.shape[1] == 3:
        # RGB to Luma (Rec.709 approximate as used in FSR often)
        luma = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
    else:
        luma = x

    # Pad for 3x3 window
    padded = F.pad(luma, (1, 1, 1, 1), mode='replicate')
    
    # Extract neighbors
    # b c d
    # e f g
    # h i j
    # where f is the center pixel (luma)
    
    # Top row
    b = padded[:, :, 0:-2, 1:-1]
    
    # Middle row
    f = padded[:, :, 1:-1, 1:-1] # center
    e = padded[:, :, 1:-1, 0:-2] # left
    g = padded[:, :, 1:-1, 2:]   # right
    
    # Bottom row
    h = padded[:, :, 2:, 1:-1]

    # FSR EASU generic edge check (simplified) determines edge presence by checking difference
    # between center and neighbors in a + shape (b, h, e, g)
    # Strength = |b-f| + |h-f| + |e-f| + |g-f|
    
    edge = torch.abs(b - f) + torch.abs(h - f) + torch.abs(e - f) + torch.abs(g - f)
    
    # Normalize edge map to (0, 1) by clamping
    return edge.clamp(0.0, 1.0)

class DIV2K_Dataset(Dataset):
    def __init__(self, hr_dir: str, lr_dir: str, scale: int = 4, patch_size: int = 96, augment: bool = True):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment

        self.hr_images = sorted(os.listdir(hr_dir))

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.hr_images)
    
    def random_crop(self, lr, hr):
        lr_w, lr_h = lr.size
        lr_patch = self.patch_size
        hr_patch = lr_patch * self.scale

        x = random.randint(0, lr_w - lr_patch)
        y = random.randint(0, lr_h - lr_patch)

        lr_crop = lr.crop((x, y, x + lr_patch, y + lr_patch))

        hr_x = x * self.scale
        hr_y = y * self.scale

        hr_crop = hr.crop(
            (hr_x, hr_y, hr_x + hr_patch, hr_y + hr_patch)
        )

        return lr_crop, hr_crop

    def augment_fn(self, lr, hr):
        """Synchronized augmentation: both images get same transforms."""
        if random.random() < 0.5:
            lr = lr.transpose(Image.FLIP_LEFT_RIGHT)
            hr = hr.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            lr = lr.transpose(Image.FLIP_TOP_BOTTOM)
            hr = hr.transpose(Image.FLIP_TOP_BOTTOM)

        return lr, hr

    def __getitem__(self, idx):

        hr_name = self.hr_images[idx]

        hr_path = os.path.join(self.hr_dir, hr_name)
        lr_path = os.path.join(
            self.lr_dir,
            hr_name.replace(".png", f"x{self.scale}.png")
        )

        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")

        lr, hr = self.random_crop(lr, hr)
        if self.augment:
            # Synchronized augmentation: both LR and HR get same transforms
            # Edge maps computed below will reflect these same spatial transforms
            lr, hr = self.augment_fn(lr, hr)

        lr = self.to_tensor(lr)
        hr = self.to_tensor(hr)

        # Use FSR edge map
        edge_lr = fsr_edge(lr.unsqueeze(0)).squeeze(0)
        edge_hr = fsr_edge(hr.unsqueeze(0)).squeeze(0)

        return lr, edge_lr, hr, edge_hr
    

class DIV2K_Validation(Dataset):
    def __init__(self, hr_dir: str, lr_dir: str, scale: int = 4):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.scale = scale
        self.to_tensor = T.ToTensor()
        self.hr_images = sorted(os.listdir(hr_dir))

    def __len__(self):
        return len(self.hr_images)
    
    def __getitem__(self, idx):
        hr_name = self.hr_images[idx]

        hr_path = os.path.join(self.hr_dir, hr_name)
        lr_path = os.path.join(
            self.lr_dir,
            hr_name.replace(".png", f"x{self.scale}.png")
        )

        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")

        lr = self.to_tensor(lr)
        hr = self.to_tensor(hr)

        # Use FSR edge map
        edge_lr = fsr_edge(lr.unsqueeze(0)).squeeze(0)
        edge_hr = fsr_edge(hr.unsqueeze(0)).squeeze(0)

        return lr, edge_lr, hr, edge_hr
        
