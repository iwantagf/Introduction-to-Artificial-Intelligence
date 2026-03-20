import argparse
import math
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.utils as vutils
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from dataset import DIV2K_Dataset, DIV2K_Validation, fsr_edge, to_grayscale
from model import EdgeGuidedCNN, apply_rcas
from copy import deepcopy


class ModelEMA:
    """ Exponential Moving Average for model parameters """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
                
    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)


def preprocess_edge(edge: torch.Tensor, scale: int) -> torch.Tensor:
	"""
	Preprocess edge map to reduce noise:
	1. Thresholding: set weak edges to 0
	2. Conditional Gaussian Blur: if edge map is too noisy (high mean intensity)
	Since edge is now (B, 2, H, W) [Gx, Gy], we use magnitude for logical checks.
	Blur strength is adjusted by the upscaling factor.
	"""
	# Check magnitude for thresholding decisions
	mag = torch.norm(edge, dim=1, keepdim=True) # [B, 1, H, W]

	# 1. Thresholding
	threshold = 0.1
	mask = (mag >= threshold).float()
	edge = edge * mask

	# 2. Conditional Gaussian Blur
	# Checks mean of magnitude
	if mag.mean() > 0.15:
		kernel_size = 5 if scale >= 4 else 3
		sigma = 0.5 * max(1.0, scale / 2.0)
		edge = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(edge)

	return edge


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
	mse = torch.mean((pred - target) ** 2)
	if torch.isnan(mse) or mse == 0:
		return torch.tensor(0.0, device=pred.device)
	return 20 * torch.log10(1.0 / torch.sqrt(mse + eps))


def rgb_to_y_channel(image: torch.Tensor) -> torch.Tensor:
	if image.size(1) == 1:
		return image

	coeffs = image.new_tensor([65.481, 128.553, 24.966]).view(1, 3, 1, 1) / 255.0
	return (image * coeffs).sum(dim=1, keepdim=True) + (16.0 / 255.0)


def shave_border(image: torch.Tensor, shave: int) -> torch.Tensor:
	if shave <= 0:
		return image
	if image.size(-2) <= 2 * shave or image.size(-1) <= 2 * shave:
		return image
	return image[..., shave:-shave, shave:-shave]


def benchmark_psnr(
	pred: torch.Tensor,
	target: torch.Tensor,
	shave: int = 0,
	y_channel: bool = True,
	eps: float = 1e-8,
) -> torch.Tensor:
	pred_eval = pred.clamp(0, 1)
	target_eval = target.clamp(0, 1)

	if y_channel:
		pred_eval = rgb_to_y_channel(pred_eval)
		target_eval = rgb_to_y_channel(target_eval)

	pred_eval = shave_border(pred_eval, shave)
	target_eval = shave_border(target_eval, shave)

	mse = torch.mean((pred_eval - target_eval) ** 2)
	if torch.isnan(mse) or mse <= eps:
		return torch.tensor(0.0, device=pred.device)
	return 20 * torch.log10(1.0 / torch.sqrt(mse + eps))


def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
	coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
	kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
	kernel_1d = kernel_1d / kernel_1d.sum()
	kernel_2d = torch.outer(kernel_1d, kernel_1d)
	return kernel_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def benchmark_ssim(
	pred: torch.Tensor,
	target: torch.Tensor,
	shave: int = 0,
	y_channel: bool = True,
	window_size: int = 11,
	sigma: float = 1.5,
	eps: float = 1e-8,
) -> torch.Tensor:
	pred_eval = pred.clamp(0, 1)
	target_eval = target.clamp(0, 1)

	if y_channel:
		pred_eval = rgb_to_y_channel(pred_eval)
		target_eval = rgb_to_y_channel(target_eval)

	pred_eval = shave_border(pred_eval, shave)
	target_eval = shave_border(target_eval, shave)

	_, channels, height, width = pred_eval.shape
	effective_window = min(window_size, height, width)
	if effective_window % 2 == 0:
		effective_window -= 1
	if effective_window < 1:
		return torch.tensor(0.0, device=pred.device)

	effective_sigma = sigma * effective_window / max(window_size, 1)
	window = _gaussian_window(
		effective_window,
		max(effective_sigma, eps),
		channels,
		pred_eval.device,
		pred_eval.dtype,
	)
	padding = effective_window // 2

	mu_pred = F.conv2d(pred_eval, window, padding=padding, groups=channels)
	mu_target = F.conv2d(target_eval, window, padding=padding, groups=channels)

	mu_pred_sq = mu_pred.pow(2)
	mu_target_sq = mu_target.pow(2)
	mu_pred_target = mu_pred * mu_target

	sigma_pred_sq = F.conv2d(pred_eval * pred_eval, window, padding=padding, groups=channels) - mu_pred_sq
	sigma_target_sq = F.conv2d(target_eval * target_eval, window, padding=padding, groups=channels) - mu_target_sq
	sigma_pred_target = F.conv2d(pred_eval * target_eval, window, padding=padding, groups=channels) - mu_pred_target

	c1 = (0.01 ** 2)
	c2 = (0.03 ** 2)
	sigma_pred_sq = torch.clamp(sigma_pred_sq, min=0.0)
	sigma_target_sq = torch.clamp(sigma_target_sq, min=0.0)

	ssim_map = (
		(2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
	) / (
		(mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2) + eps
	)
	return ssim_map.mean()


def save_preview(lr, sr, hr, save_path: Path, scale: int):
    """Save SR and HR as separate PNG files."""
    target_size = hr.shape[-2:]
    sr_resized = F.interpolate(sr, size=target_size, mode="bicubic", align_corners=False)
    
    # Move to CPU and handle NaN values
    sr_resized = sr_resized.cpu().detach().clamp(0, 1).float()
    hr = hr.cpu().detach().clamp(0, 1).float()
    
    # Replace NaN with 0
    sr_resized = torch.where(torch.isnan(sr_resized), torch.zeros_like(sr_resized), sr_resized)
    hr = torch.where(torch.isnan(hr), torch.zeros_like(hr), hr)
    
    to_pil = T.ToPILImage()
    sr_pil = to_pil(sr_resized.squeeze(0))
    hr_pil = to_pil(hr.squeeze(0))
    
    # Save SR and HR separately
    save_path_str = str(save_path).replace(".png", "")
    sr_pil.save(f"{save_path_str}_sr.png")
    hr_pil.save(f"{save_path_str}_hr.png")

def build_loaders(scale: int, batch_size: int, num_workers: int, lr_mode: str, val_batch_size: int = 1, patch_size: int = 192):
	hr_train = "./DIV2K/DIV2K_train_HR"
	hr_valid = "./DIV2K/DIV2K_valid_HR"

	if lr_mode == "mixed":
		# Load both bicubic and unknown datasets
		lr_train_bicubic = f"./DIV2K/DIV2K_train_LR_bicubic/X{scale}"
		lr_valid_bicubic = f"./DIV2K/DIV2K_valid_LR_bicubic/X{scale}"
		
		lr_train_unknown = f"./DIV2K/DIV2K_train_LR_unknown/X{scale}"
		lr_valid_unknown = f"./DIV2K/DIV2K_valid_LR_unknown/X{scale}"

		train_dataset_bicubic = DIV2K_Dataset(hr_dir=hr_train, lr_dir=lr_train_bicubic, scale=scale, patch_size=patch_size, augment=True)
		train_dataset_unknown = DIV2K_Dataset(hr_dir=hr_train, lr_dir=lr_train_unknown, scale=scale, patch_size=patch_size, augment=True)
		
		valid_dataset_bicubic = DIV2K_Validation(hr_dir=hr_valid, lr_dir=lr_valid_bicubic, scale=scale)
		valid_dataset_unknown = DIV2K_Validation(hr_dir=hr_valid, lr_dir=lr_valid_unknown, scale=scale)
		
		# Concatenate datasets
		train_dataset = ConcatDataset([train_dataset_bicubic, train_dataset_unknown])
		valid_dataset = ConcatDataset([valid_dataset_bicubic, valid_dataset_unknown])
		
	elif lr_mode == "bicubic":
		lr_train = f"./DIV2K/DIV2K_train_LR_bicubic/X{scale}"
		lr_valid = f"./DIV2K/DIV2K_valid_LR_bicubic/X{scale}"
		train_dataset = DIV2K_Dataset(hr_dir=hr_train, lr_dir=lr_train, scale=scale, patch_size=patch_size, augment=True)
		valid_dataset = DIV2K_Validation(hr_dir=hr_valid, lr_dir=lr_valid, scale=scale)
	else:
		lr_train = f"./DIV2K/DIV2K_train_LR_unknown/X{scale}"
		lr_valid = f"./DIV2K/DIV2K_valid_LR_unknown/X{scale}"
		train_dataset = DIV2K_Dataset(hr_dir=hr_train, lr_dir=lr_train, scale=scale, patch_size=patch_size, augment=True)
		valid_dataset = DIV2K_Validation(hr_dir=hr_valid, lr_dir=lr_valid, scale=scale)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
	valid_loader = DataLoader(valid_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
	return train_loader, valid_loader


def train_one_epoch(model, loader, optimizer, scaler, device, lambda_edge: float, scale: int, ema=None, use_rcas: bool = False, rcas_strength: float = 0.3):
	model.train()
	criterion = nn.MSELoss()
	running_loss = 0.0

	for lr, edge_lr, hr, edge_hr in tqdm(loader, desc=f"Train (λ_edge={lambda_edge:.4f})", leave=False):
		lr = lr.to(device)
		edge_lr = edge_lr.to(device)
		hr = hr.to(device)
		edge_hr = edge_hr.to(device)

		# Preprocess edge map
		edge_lr = preprocess_edge(edge_lr, scale)

		optimizer.zero_grad(set_to_none=True)

		with amp.autocast(enabled=device.type == "cuda"):
			sr = model(lr, edge_lr)
			lr_upsampled = F.interpolate(lr, size=sr.shape[-2:], mode='bicubic', align_corners=False).clamp(0, 1)
			sr = (sr + lr_upsampled).clamp(0, 1)
			
			# Ensure no NaNs in SR before loss computation
			if torch.isnan(sr).any():
				optimizer.zero_grad(set_to_none=True)
				continue

			# Compute edge map from SR output (grayscale using standard luminance formula)
			sr_gray = to_grayscale(sr)
			sr_edge = fsr_edge(sr_gray)
			
			# Sanity check for loss inputs
			if torch.isnan(sr).any() or torch.isinf(sr).any():
				optimizer.zero_grad(set_to_none=True)
				continue

			# Loss: RGB reconstruction loss + edge alignment loss with HR edge target
			loss_rgb = criterion(sr.clamp(0, 1), hr)
			loss_edge = criterion(sr_edge, edge_hr)
			loss = loss_rgb + lambda_edge * loss_edge

			# Apply RCAS *after* calculating loss (for visual quality, but not optimization) if needed in future
			# Currently not used in training for visualization, but logic requested.
			# if use_rcas:
			# 	sr = apply_rcas(sr, strength=rcas_strength).clamp(0, 1)

		if torch.isnan(loss) or torch.isinf(loss):
			# print(f"Warning: NaN loss detected at epoch {epoch}. Skipping optimization step.")
			optimizer.zero_grad(set_to_none=True)
			scaler.update()
			continue

		scaler.scale(loss).backward()
		scaler.unscale_(optimizer)
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
		scaler.step(optimizer)
		scaler.update()

		if ema:
			ema.update(model)

		running_loss += loss.item()

	return running_loss / max(1, len(loader))


def validate(model, loader, device, lambda_edge: float, scale: int, preview_dir: Path, preview_count: int, epoch: int, use_rcas: bool = False, rcas_strength: float = 0.3):
	model.eval()
	criterion = nn.MSELoss()
	total_loss = 0.0
	total_psnr = 0.0
	total_benchmark_ssim = 0.0
	total_lr_psnr = 0.0  # Baseline: bicubic LR upsampled PSNR
	total_lr_benchmark_ssim = 0.0
	
	total_samples = len(loader.dataset)
	# Pick random start index for previews
	start_preview_idx = random.randint(0, max(0, total_samples - preview_count))
	current_idx = 0
	saved = 0

	with torch.no_grad():
		for batch_idx, (lr, edge_lr, hr, edge_hr) in enumerate(tqdm(loader, desc="Valid", leave=False)):
			lr = lr.to(device)
			edge_lr = edge_lr.to(device)
			# Preprocess edge map
			edge_lr = preprocess_edge(edge_lr, scale)

			hr = hr.to(device)
			edge_hr = edge_hr.to(device)

			with amp.autocast(enabled=device.type == "cuda"):
				sr = model(lr, edge_lr)
				lr_upsampled = F.interpolate(lr, size=sr.shape[-2:], mode='bicubic', align_corners=False).clamp(0, 1)
				sr = (sr + lr_upsampled).clamp(0, 1)
				
				sr_gray = to_grayscale(sr)
				sr_edge = fsr_edge(sr_gray)
				loss_rgb = criterion(sr.clamp(0, 1), hr)
				loss_edge = criterion(sr_edge, edge_hr)
				loss = loss_rgb + lambda_edge * loss_edge

				# Compute bicubic LR upsampled PSNR for baseline comparison
				lr_upsampled_psnr = F.interpolate(lr, size=hr.shape[-2:], mode='bicubic', align_corners=False).clamp(0, 1)

			if torch.isnan(loss):
				continue

			total_loss += loss.item()
			total_psnr += psnr(sr.clamp(0, 1), hr).item()
			total_benchmark_ssim += benchmark_ssim(sr, hr, shave=scale, y_channel=True).item()
			total_lr_psnr += psnr(lr_upsampled_psnr, hr).item()
			total_lr_benchmark_ssim += benchmark_ssim(lr_upsampled_psnr, hr, shave=scale, y_channel=True).item()

			# Metrics are computed on raw model output; RCAS is preview-only.
			if use_rcas:
				sr = apply_rcas(sr, strength=rcas_strength).clamp(0, 1)

			# Random continuous preview saving logic
			batch_size = lr.size(0)
			if saved < preview_count and current_idx + batch_size > start_preview_idx:
				start_in_batch = max(0, start_preview_idx - current_idx)
				end_in_batch = min(batch_size, start_preview_idx + preview_count - current_idx)
				
				for i in range(start_in_batch, end_in_batch):
					if saved >= preview_count: break
					save_path = preview_dir / f"preview_{saved}.png"
					save_preview(lr[i:i+1].cpu(), sr[i:i+1].cpu(), hr[i:i+1].cpu(), save_path, scale)
					saved += 1
			
			current_idx += batch_size

	n = max(1, len(loader))
	return total_loss / n, total_psnr / n, total_benchmark_ssim / n, total_lr_psnr / n, total_lr_benchmark_ssim / n


def save_checkpoint(state: dict, is_best: bool, save_dir: Path):
	save_dir.mkdir(parents=True, exist_ok=True)
	last_path = save_dir / "last.pth"
	torch.save(state, last_path)
	if is_best:
		best_path = save_dir / "best.pth"
		torch.save(state, best_path)


def main():
	parser = argparse.ArgumentParser(description="Edge-Guided SR Training")
	parser.add_argument("--epochs", type=int, default=500)
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
	parser.add_argument("--lambda_edge", type=float, default=0.015)
	parser.add_argument("--scale", type=int, default=4)
	parser.add_argument("--num_workers", type=int, default=6)
	parser.add_argument("--lr_mode", type=str, choices=["bicubic", "unknown", "mixed"], default="mixed")
	parser.add_argument("--save_dir", type=str, default="outputs")
	parser.add_argument("--preview_count", type=int, default=3)
	parser.add_argument("--resume", type=str, default=None)
	parser.add_argument("--val_batch_size", type=int, default=1)
	parser.add_argument("--patch_size", type=int, default=96)
	parser.add_argument("--use_rcas", type=int, default=1, help="apply RCAS sharpening (0 or 1)")
	parser.add_argument("--rcas_strength", type=float, default=0.05, help="RCAS sharpening strength")
	parser.add_argument("--use_edge_branch", type=int, default=1, help="use separate edge branch (0 or 1)")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	torch.backends.cudnn.benchmark = True

	save_dir = Path(args.save_dir)
	ckpt_dir = save_dir / "checkpoints"
	preview_dir = save_dir / "preview"
	preview_dir.mkdir(parents=True, exist_ok=True)

	train_loader, valid_loader = build_loaders(args.scale, args.batch_size, args.num_workers, args.lr_mode, args.val_batch_size, args.patch_size)

	# Lightweight model structure for weak devices (approx FSR complexity)
	model = EdgeGuidedCNN(input_channels=4, num_features=64, head_features=48, scale=args.scale, num_blocks=3, use_edge_branch=bool(args.use_edge_branch)).to(device)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
		optimizer,
		T_0=50,
		T_mult=2,
		eta_min=1e-7,
	)

	scaler = amp.GradScaler(enabled=device.type == "cuda")

	# Initialize EMA
	ema = ModelEMA(model, decay=0.999)

	start_epoch = 0
	best_metric = -1e9

	if args.resume and os.path.isfile(args.resume):
		checkpoint = torch.load(args.resume, map_location=device)
		model.load_state_dict(checkpoint.get("model", checkpoint))
		optimizer.load_state_dict(checkpoint["optimizer"])
		scaler.load_state_dict(checkpoint["scaler"])
		start_epoch = checkpoint.get("epoch", 0)
		best_metric = checkpoint.get("best_ssim", checkpoint.get("best_metric", checkpoint.get("best_psnr", best_metric)))
		if "ema" in checkpoint:
			ema.load_state_dict(checkpoint["ema"])

	initial_lambda_edge = args.lambda_edge
	peak_lambda_edge = 0.05
	final_lambda_edge = 0.001  # Decay to almost zero to focus on pixel fidelity

	for epoch in range(start_epoch, args.epochs):
		# 3-phase lambda_edge schedule
		progress = epoch / args.epochs
		if progress < 0.3:
			# Phase 1: Warmup to peak (0-30%)
			phase_p = progress / 0.3
			current_lambda = initial_lambda_edge + (peak_lambda_edge - initial_lambda_edge) * phase_p
		elif progress < 0.7:
			# Phase 2: Constant peak (30-70%)
			current_lambda = peak_lambda_edge
		else:
			# Phase 3: Decay to final (70-100%) - focus on pixel fidelity
			phase_p = (progress - 0.7) / 0.3
			current_lambda = peak_lambda_edge - (peak_lambda_edge - final_lambda_edge) * phase_p
		
		train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, current_lambda, args.scale, ema, bool(args.use_rcas), args.rcas_strength)
		val_loss, val_psnr, val_benchmark_ssim, lr_psnr, lr_benchmark_ssim = validate(model, valid_loader, device, current_lambda, args.scale, preview_dir, args.preview_count, epoch, bool(args.use_rcas), args.rcas_strength)

		scheduler.step(epoch + 1)

		state = {
			"epoch": epoch + 1,
			"model": model.state_dict(),
			"ema": ema.state_dict(),
			"optimizer": optimizer.state_dict(),
			"scaler": scaler.state_dict(),
			"best_ssim": best_metric,
		}

		is_best = val_benchmark_ssim > best_metric
		if is_best:
			best_metric = val_benchmark_ssim
			print(f"New best benchmark SSIM: {best_metric:.4f} at epoch {epoch+1} with λ_edge: {current_lambda:.4f}")
			print(f"Saving best checkpoint to {ckpt_dir / 'best.pth'}")
		save_checkpoint(state, is_best, ckpt_dir)

		print(
			f"Epoch {epoch+1}/{args.epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} "
			f"| val_psnr_rgb: {val_psnr:.2f} | val_ssim_y: {val_benchmark_ssim:.4f} "
			f"| lr_psnr_rgb: {lr_psnr:.2f} | lr_ssim_y: {lr_benchmark_ssim:.4f}"
		)


if __name__ == "__main__":
	main()
