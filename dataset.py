import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import os
import random


def _list_image_files(image_dir: str):
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted(
        file_name
        for file_name in os.listdir(image_dir)
        if os.path.splitext(file_name)[1].lower() in valid_exts
    )


def _resolve_lr_filename(hr_name: str, lr_dir: str, scale: int) -> str:
    stem, ext = os.path.splitext(hr_name)
    candidates = [
        hr_name,
        f"{stem}x{scale}{ext}",
        f"{stem}_x{scale}{ext}",
        f"{stem}X{scale}{ext}",
        f"{stem}_LRBI_x{scale}{ext}",
        f"{stem}_LRB_x{scale}{ext}",
        f"{stem}_LR_x{scale}{ext}",
    ]

    for candidate in candidates:
        if os.path.exists(os.path.join(lr_dir, candidate)):
            return candidate

    raise FileNotFoundError(
        f"Could not find LR pair for '{hr_name}' in '{lr_dir}' with scale x{scale}."
    )


def _modcrop_hr_to_match_lr(hr: Image.Image, lr: Image.Image, scale: int) -> Image.Image:
    expected_width = lr.width * scale
    expected_height = lr.height * scale

    if hr.width == expected_width and hr.height == expected_height:
        return hr

    crop_width = min(hr.width, expected_width)
    crop_height = min(hr.height, expected_height)
    return hr.crop((0, 0, crop_width, crop_height))


# Edge Extraction RCAS
def to_grayscale(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB to grayscale using standard luminance formula: 0.299R + 0.587G + 0.114B."""
    # rgb shape: (B, 3, H, W)
    return 0.299 * rgb[:, 0:1, :, :] + 0.587 * rgb[:, 1:2, :, :] + 0.114 * rgb[:, 2:3, :, :]


def fsr_edge(x):
    """
    Compute edge map using Sobel gradients to capture directionality.
    Returns 2-channel edge map (Gx, Gy) normalized vaguely to [-1, 1].
    """
    if x.shape[1] == 3:
        # RGB to Luma (Rec.709 approximate as used in FSR often)
        luma = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
    else:
        luma = x

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device).view(1, 1, 3, 3).float()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device).view(1, 1, 3, 3).float()
    
    gx = F.conv2d(luma, sobel_x, padding=1)
    gy = F.conv2d(luma, sobel_y, padding=1)
    
    # We use tanh to normalize to (-1, 1) approximately to handle varying intensities
    return torch.cat([torch.tanh(gx), torch.tanh(gy)], dim=1)

class DIV2K_Dataset(Dataset):
    def __init__(self, hr_dir: str, lr_dir: str, scale: int = 4, patch_size: int = 96, augment: bool = True):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment

        self.hr_images = _list_image_files(hr_dir)

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
        self.hr_images = _list_image_files(hr_dir)

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


class BenchmarkDataset(Dataset):
    def __init__(self, hr_dir: str, lr_dir: str, scale: int = 4):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.scale = scale
        self.to_tensor = T.ToTensor()
        self.hr_images = _list_image_files(hr_dir)

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_name = self.hr_images[idx]
        lr_name = _resolve_lr_filename(hr_name, self.lr_dir, self.scale)

        hr_path = os.path.join(self.hr_dir, hr_name)
        lr_path = os.path.join(self.lr_dir, lr_name)

        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")

        hr = _modcrop_hr_to_match_lr(hr, lr, self.scale)

        lr = self.to_tensor(lr)
        hr = self.to_tensor(hr)

        edge_lr = fsr_edge(lr.unsqueeze(0)).squeeze(0)
        edge_hr = fsr_edge(hr.unsqueeze(0)).squeeze(0)

        return lr, edge_lr, hr, edge_hr
        
