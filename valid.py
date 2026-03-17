import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda import amp

from dataset import DIV2K_Validation, fsr_edge, to_grayscale
from train import build_loaders, save_preview, psnr
from model import EdgeGuidedCNN, apply_rcas


def run_validation(checkpoint: str, scale: int, batch_size: int, num_workers: int, lr_mode: str, preview_count: int, save_dir: str, lambda_edge: float = 0.05, use_rcas: bool = True, rcas_strength: float = 0.05):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	preview_dir = Path(save_dir) / "preview"
	preview_dir.mkdir(parents=True, exist_ok=True)

	_, valid_loader = build_loaders(scale, batch_size, num_workers, lr_mode, val_batch_size=batch_size)

	# Lightweight model structure matching train.py
	model = EdgeGuidedCNN(input_channels=4, num_features=64, head_features=48, scale=scale, num_blocks=16).to(device)
	checkpoint_data = torch.load(checkpoint, map_location=device)
	model.load_state_dict(checkpoint_data.get("model", checkpoint_data))
	model.eval()

	criterion = torch.nn.MSELoss()
	total_loss = 0.0
	total_psnr = 0.0
	total_lr_psnr = 0.0  # Baseline: bicubic LR upsampled PSNR
	saved = 0

	with torch.no_grad():
		for batch_idx, (lr, edge_lr, hr, edge_hr) in enumerate(valid_loader):
			lr = lr.to(device)
			edge_lr = edge_lr.to(device)
			hr = hr.to(device)
			edge_hr = edge_hr.to(device)

			with amp.autocast(enabled=device.type == "cuda"):
				sr = model(lr, edge_lr)
				# Add global residual: enhance upsampled LR
				lr_upsampled = F.interpolate(lr, size=sr.shape[-2:], mode='bicubic', align_corners=False).clamp(0, 1)
				sr = (sr + lr_upsampled).clamp(0, 1)  # Clamp after addition
				
				sr_gray = to_grayscale(sr)
				sr_edge = fsr_edge(sr_gray)
				loss_rgb = criterion(sr.clamp(0, 1), hr)
				loss_edge = criterion(sr_edge, edge_hr)
				loss = loss_rgb + lambda_edge * loss_edge

				# Compute bicubic LR upsampled PSNR for baseline comparison
				lr_upsampled_psnr = F.interpolate(lr, size=hr.shape[-2:], mode='bicubic', align_corners=False).clamp(0, 1)

			total_loss += loss.item()
			total_psnr += psnr(sr.clamp(0, 1), hr).item()
			total_lr_psnr += psnr(lr_upsampled_psnr, hr).item()

			# Sharpen image AFTER loss calculation for preview
			if use_rcas:
				sr = apply_rcas(sr, strength=rcas_strength).clamp(0, 1)

			if saved < preview_count:
				for i in range(min(preview_count - saved, lr.size(0))):
					save_path = preview_dir / f"eval_idx_{batch_idx:03d}_{i}.png"
					save_preview(lr[i:i+1].cpu(), sr[i:i+1].cpu(), hr[i:i+1].cpu(), save_path, scale)
					saved += 1

	n = max(1, len(valid_loader))
	avg_loss = total_loss / n
	avg_psnr = total_psnr / n
	avg_lr_psnr = total_lr_psnr / n
	print(f"Validation | loss: {avg_loss:.4f} | model_psnr: {avg_psnr:.2f} | lr_psnr: {avg_lr_psnr:.2f}")


def main():
	parser = argparse.ArgumentParser(description="Validate SR checkpoint")
	parser.add_argument("--checkpoint", type=str, required=True)
	parser.add_argument("--scale", type=int, default=4)
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--num_workers", type=int, default=6)
	parser.add_argument("--lr_mode", type=str, choices=["bicubic", "unknown", "mixed"], default="mixed")
	parser.add_argument("--preview_count", type=int, default=3)
	parser.add_argument("--save_dir", type=str, default="outputs")
	parser.add_argument("--lambda_edge", type=float, default=0.05)
	args = parser.parse_args()

	run_validation(
		checkpoint=args.checkpoint,
		scale=args.scale,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		lr_mode=args.lr_mode,
		preview_count=args.preview_count,
		save_dir=args.save_dir,
		lambda_edge=args.lambda_edge,
	)


if __name__ == "__main__":
	main()
