import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda import amp
from tqdm import tqdm

from dataset import BenchmarkDataset, fsr_edge, to_grayscale
from train import benchmark_psnr, benchmark_ssim, build_loaders, save_preview, preprocess_edge
from model import EdgeGuidedCNN, apply_rcas


def load_model(checkpoint: str, scale: int, device: torch.device) -> EdgeGuidedCNN:
	model = EdgeGuidedCNN(input_channels=4, num_features=64, head_features=48, scale=scale, num_blocks=3).to(device)
	checkpoint_data = torch.load(checkpoint, map_location=device)
	state_dict = checkpoint_data.get("ema", checkpoint_data.get("model", checkpoint_data))
	model.load_state_dict(state_dict)
	model.eval()
	return model


def crop_tensor_pair(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	if pred.shape[-2:] == target.shape[-2:]:
		return pred, target

	height = min(pred.shape[-2], target.shape[-2])
	width = min(pred.shape[-1], target.shape[-1])
	return pred[..., :height, :width], target[..., :height, :width]


def run_validation_loader(model, loader, device, lambda_edge: float, scale: int, preview_dir: Path, preview_count: int, use_rcas: bool = True, rcas_strength: float = 0.05, dataset_name: str = "validation"):
	criterion = torch.nn.MSELoss()
	total_loss = 0.0
	total_ssim_y = 0.0
	total_psnr_y = 0.0
	total_lr_ssim_y = 0.0
	total_lr_psnr_y = 0.0

	ssim_y_list = []
	psnr_y_list = []
	lr_ssim_y_list = []
	lr_psnr_y_list = []
	saved = 0

	with torch.no_grad():
		for batch_idx, (lr, edge_lr, hr, edge_hr) in enumerate(tqdm(loader, desc=f"Valid {dataset_name}", leave=False)):
			lr = lr.to(device)
			edge_lr = edge_lr.to(device)
			hr = hr.to(device)
			edge_hr = edge_hr.to(device)

			edge_lr = preprocess_edge(edge_lr, scale)

			with amp.autocast(enabled=device.type == "cuda"):
				sr = model(lr, edge_lr)
				lr_upsampled = F.interpolate(lr, size=sr.shape[-2:], mode='bicubic', align_corners=False).clamp(0, 1)
				sr = (sr + lr_upsampled).clamp(0, 1)
				sr, hr = crop_tensor_pair(sr, hr)

				sr_gray = to_grayscale(sr)
				sr_edge = fsr_edge(sr_gray)
				sr_edge, edge_hr = crop_tensor_pair(sr_edge, edge_hr)
				loss_rgb = criterion(sr.clamp(0, 1), hr)
				loss_edge = criterion(sr_edge, edge_hr)
				loss = loss_rgb + lambda_edge * loss_edge

				lr_upsampled_psnr = F.interpolate(lr, size=hr.shape[-2:], mode='bicubic', align_corners=False).clamp(0, 1)

			total_loss += loss.item()

			current_ssim_y = benchmark_ssim(sr, hr, shave=scale, y_channel=True).item()
			current_psnr_y = benchmark_psnr(sr, hr, shave=scale, y_channel=True).item()
			current_lr_ssim_y = benchmark_ssim(lr_upsampled_psnr, hr, shave=scale, y_channel=True).item()
			current_lr_psnr_y = benchmark_psnr(lr_upsampled_psnr, hr, shave=scale, y_channel=True).item()
			total_ssim_y += current_ssim_y
			total_psnr_y += current_psnr_y
			total_lr_ssim_y += current_lr_ssim_y
			total_lr_psnr_y += current_lr_psnr_y
			ssim_y_list.append(current_ssim_y)
			psnr_y_list.append(current_psnr_y)
			lr_ssim_y_list.append(current_lr_ssim_y)
			lr_psnr_y_list.append(current_lr_psnr_y)

			if use_rcas:
				sr = apply_rcas(sr, strength=rcas_strength).clamp(0, 1)

			if saved < preview_count:
				for i in range(min(preview_count - saved, lr.size(0))):
					save_path = preview_dir / f"{dataset_name}_idx_{batch_idx:03d}_{i}.png"
					save_preview(lr[i:i+1].cpu(), sr[i:i+1].cpu(), hr[i:i+1].cpu(), save_path, scale)
					saved += 1

	n = max(1, len(loader))
	results = {
		"dataset": dataset_name,
		"loss": total_loss / n,
		"avg_ssim_y": total_ssim_y / n,
		"avg_psnr_y": total_psnr_y / n,
		"avg_bicubic_ssim_y": total_lr_ssim_y / n,
		"avg_bicubic_psnr_y": total_lr_psnr_y / n,
		"min_ssim_y": min(ssim_y_list) if ssim_y_list else 0.0,
		"max_ssim_y": max(ssim_y_list) if ssim_y_list else 0.0,
		"min_psnr_y": min(psnr_y_list) if psnr_y_list else 0.0,
		"max_psnr_y": max(psnr_y_list) if psnr_y_list else 0.0,
		"min_bicubic_ssim_y": min(lr_ssim_y_list) if lr_ssim_y_list else 0.0,
		"max_bicubic_ssim_y": max(lr_ssim_y_list) if lr_ssim_y_list else 0.0,
		"min_bicubic_psnr_y": min(lr_psnr_y_list) if lr_psnr_y_list else 0.0,
		"max_bicubic_psnr_y": max(lr_psnr_y_list) if lr_psnr_y_list else 0.0,
		"samples": len(loader.dataset),
	}
	return results


def print_results(results: dict, scale: int, preview_dir: Path):
	print(f"\n=== Validation Results: {results['dataset']} ===")
	print(f"Loss: {results['loss']:.4f}")
	print(f"Average SSIM (Y, shave={scale}): {results['avg_ssim_y']:.4f}")
	print(f"Average PSNR (Y, shave={scale}): {results['avg_psnr_y']:.2f}")
	print(f"Bicubic SSIM (Y, shave={scale}): {results['avg_bicubic_ssim_y']:.4f}")
	print(f"Bicubic PSNR (Y, shave={scale}): {results['avg_bicubic_psnr_y']:.2f}")
	print(f"Min SSIM (Y, shave={scale}): {results['min_ssim_y']:.4f}")
	print(f"Max SSIM (Y, shave={scale}): {results['max_ssim_y']:.4f}")
	print(f"Min PSNR (Y, shave={scale}): {results['min_psnr_y']:.2f}")
	print(f"Max PSNR (Y, shave={scale}): {results['max_psnr_y']:.2f}")
	print(f"Min Bicubic SSIM (Y, shave={scale}): {results['min_bicubic_ssim_y']:.4f}")
	print(f"Max Bicubic SSIM (Y, shave={scale}): {results['max_bicubic_ssim_y']:.4f}")
	print(f"Min Bicubic PSNR (Y, shave={scale}): {results['min_bicubic_psnr_y']:.2f}")
	print(f"Max Bicubic PSNR (Y, shave={scale}): {results['max_bicubic_psnr_y']:.2f}")
	print(f"Previews saved to: {preview_dir}")


def resolve_benchmark_paths(benchmark_root: Path, dataset_name: str, scale: int):
	hr_dir = benchmark_root / dataset_name / "HR"
	lr_dir = benchmark_root / dataset_name / "LR_bicubic" / f"X{scale}"
	return hr_dir, lr_dir


def run_benchmark_suite(model, benchmark_root: str, dataset_names: list[str], scale: int, device: torch.device, lambda_edge: float, preview_dir: Path, preview_count: int, use_rcas: bool, rcas_strength: float):
	results = []
	for dataset_name in dataset_names:
		hr_dir, lr_dir = resolve_benchmark_paths(Path(benchmark_root), dataset_name, scale)
		if not hr_dir.exists() or not lr_dir.exists():
			print(f"[Skip] {dataset_name}: missing paths {hr_dir} or {lr_dir}")
			continue

		dataset = BenchmarkDataset(str(hr_dir), str(lr_dir), scale=scale)
		loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
		dataset_preview_dir = preview_dir / dataset_name
		dataset_preview_dir.mkdir(parents=True, exist_ok=True)
		result = run_validation_loader(
			model,
			loader,
			device,
			lambda_edge,
			scale,
			dataset_preview_dir,
			preview_count,
			use_rcas,
			rcas_strength,
			dataset_name,
		)
		print_results(result, scale, dataset_preview_dir)
		results.append(result)
	return results


def run_validation(checkpoint: str, scale: int, batch_size: int, num_workers: int, lr_mode: str, preview_count: int, save_dir: str, lambda_edge: float = 0.05, use_rcas: bool = True, rcas_strength: float = 0.05):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	preview_dir = Path(save_dir) / "preview"
	preview_dir.mkdir(parents=True, exist_ok=True)

	# Validation loader must use batch_size=1 because validation images have different sizes
	_, valid_loader = build_loaders(scale, 1, num_workers, lr_mode, val_batch_size=1)

	# Lightweight model structure matching train.py
	try:
		model = load_model(checkpoint, scale, device)
	except RuntimeError as e:
		print(f"\n[Error] Checkpoint loading failed: {e}")
		print("[Info] The model architecture has changed (new gradients inputs, gate mechanisms).")
		print("[Info] Please retrain the model using 'python main.py' to generate a compatible checkpoint.")
		return

	results = run_validation_loader(
		model,
		valid_loader,
		device,
		lambda_edge,
		scale,
		preview_dir,
		preview_count,
		use_rcas,
		rcas_strength,
		"DIV2K",
	)
	print_results(results, scale, preview_dir)
 

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
	parser.add_argument("--benchmark_root", type=str, default="./benchmarks")
	parser.add_argument(
		"--benchmark_sets",
		nargs="+",
		default=["Set5", "Set14", "Urban100", "Manga109", "B100"],
		help="Benchmark datasets to validate. Expected layout: <root>/<set>/HR and <root>/<set>/LR_bicubic/X<scale>",
	)
	parser.add_argument("--run_benchmark_suite", action="store_true", help="Run best.pth on classic benchmark datasets")
	parser.add_argument("--save_json", type=str, default=None, help="Optional JSON path for benchmark suite results")
	args = parser.parse_args()

	if args.run_benchmark_suite:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		preview_dir = Path(args.save_dir) / "preview"
		preview_dir.mkdir(parents=True, exist_ok=True)
		try:
			model = load_model(args.checkpoint, args.scale, device)
		except RuntimeError as e:
			print(f"\n[Error] Checkpoint loading failed: {e}")
			print("[Info] The model architecture has changed (new gradients inputs, gate mechanisms).")
			print("[Info] Please retrain the model using 'python main.py' to generate a compatible checkpoint.")
			return

		results = run_benchmark_suite(
			model,
			args.benchmark_root,
			args.benchmark_sets,
			args.scale,
			device,
			args.lambda_edge,
			preview_dir,
			args.preview_count,
			True,
			0.05,
		)
		if args.save_json and results:
			output_path = Path(args.save_json)
			output_path.parent.mkdir(parents=True, exist_ok=True)
			output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
			print(f"Saved benchmark suite JSON to: {output_path}")
		return

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
