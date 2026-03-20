import argparse
import importlib
import importlib.util
import json
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from dataset import DIV2K_Validation
from fsr_benchmark import fsr_edge_guided_upscale
from model import EdgeGuidedCNN
from train import benchmark_psnr, benchmark_ssim, preprocess_edge, rgb_to_y_channel


@dataclass
class BenchmarkResult:
	name: str
	checkpoint: str
	params: int
	avg_y_ssim: float
	avg_y_psnr: float
	avg_runtime_ms: float
	samples: int


@dataclass
class ModelSpec:
	name: str
	checkpoint: Optional[Path]
	loader: Callable[["BenchmarkContext", Optional[Path]], object]
	infer: Callable[[object, torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor]


class BenchmarkContext:
	def __init__(self, scale: int, device: torch.device, external_root: Path):
		self.scale = scale
		self.device = device
		self.external_root = external_root


class SRCNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4)
		self.conv2 = nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2)
		self.conv3 = nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = F.relu(self.conv1(x), inplace=True)
		x = F.relu(self.conv2(x), inplace=True)
		return self.conv3(x)


def count_parameters(model) -> int:
	if model is None:
		return 0
	return sum(param.numel() for param in model.parameters())


def load_state_dict_maybe_nested(checkpoint_path: Path):
	checkpoint = torch.load(checkpoint_path, map_location="cpu")
	if isinstance(checkpoint, dict):
		for key in ("ema", "params_ema", "model", "state_dict", "params"):
			if key in checkpoint and isinstance(checkpoint[key], dict):
				checkpoint = checkpoint[key]
				break

	cleaned = {}
	for key, value in checkpoint.items():
		cleaned[key[7:] if key.startswith("module.") else key] = value
	return cleaned


def load_builtin(_context: BenchmarkContext, _checkpoint: Optional[Path]):
	return None


def infer_bicubic(_model, lr: torch.Tensor, _edge_lr: torch.Tensor, hr: torch.Tensor, _scale: int) -> torch.Tensor:
	return F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)


def infer_fsr(_model, lr: torch.Tensor, _edge_lr: torch.Tensor, hr: torch.Tensor, scale: int) -> torch.Tensor:
	sr = fsr_edge_guided_upscale(lr, scale)
	return F.interpolate(sr, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)


def load_local_model(context: BenchmarkContext, checkpoint: Optional[Path]):
	if checkpoint is None or not checkpoint.exists():
		raise FileNotFoundError(f"Missing local checkpoint: {checkpoint}")

	model = EdgeGuidedCNN(
		input_channels=4,
		num_features=64,
		head_features=48,
		scale=context.scale,
		num_blocks=3,
		use_edge_branch=True,
	).to(context.device)
	model.load_state_dict(load_state_dict_maybe_nested(checkpoint), strict=True)
	model.eval()
	return model


def infer_local(model, lr: torch.Tensor, edge_lr: torch.Tensor, _hr: torch.Tensor, scale: int) -> torch.Tensor:
	edge_lr = preprocess_edge(edge_lr, scale)
	sr = model(lr, edge_lr)
	lr_upsampled = F.interpolate(lr, size=sr.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
	return (sr + lr_upsampled).clamp(0, 1)


def load_imdn_package(repo_root: Path):
	package_name = "_benchmark_imdn"
	model_dir = repo_root / "model"
	if package_name in sys.modules:
		return sys.modules[f"{package_name}.architecture"]

	package = types.ModuleType(package_name)
	package.__path__ = [str(model_dir)]
	sys.modules[package_name] = package

	block_spec = importlib.util.spec_from_file_location(f"{package_name}.block", model_dir / "block.py")
	block_module = importlib.util.module_from_spec(block_spec)
	sys.modules[f"{package_name}.block"] = block_module
	block_spec.loader.exec_module(block_module)

	arch_spec = importlib.util.spec_from_file_location(f"{package_name}.architecture", model_dir / "architecture.py")
	arch_module = importlib.util.module_from_spec(arch_spec)
	sys.modules[f"{package_name}.architecture"] = arch_module
	arch_spec.loader.exec_module(arch_module)
	return arch_module


def load_imdn_model(context: BenchmarkContext, checkpoint: Optional[Path]):
	if checkpoint is None or not checkpoint.exists():
		raise FileNotFoundError(f"Missing IMDN checkpoint: {checkpoint}")

	arch_module = load_imdn_package(context.external_root / "IMDN")
	model = arch_module.IMDN(upscale=context.scale).to(context.device)
	model.load_state_dict(load_state_dict_maybe_nested(checkpoint), strict=True)
	model.eval()
	return model


def infer_single_input(model, lr: torch.Tensor, _edge_lr: torch.Tensor, _hr: torch.Tensor, _scale: int) -> torch.Tensor:
	return model(lr).clamp(0, 1)


def load_rfdn_model(context: BenchmarkContext, checkpoint: Optional[Path]):
	if checkpoint is None or not checkpoint.exists():
		raise FileNotFoundError(f"Missing RFDN checkpoint: {checkpoint}")

	repo_root = context.external_root / "RFDN"
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))

	rfdn_module = importlib.import_module("RFDN")
	model = rfdn_module.RFDN().to(context.device)
	model.load_state_dict(load_state_dict_maybe_nested(checkpoint), strict=True)
	model.eval()
	return model


def load_srcnn_model(context: BenchmarkContext, checkpoint: Optional[Path]):
	if checkpoint is None or not checkpoint.exists():
		raise FileNotFoundError(
			"Missing SRCNN checkpoint. Expected a file like "
			f"'{checkpoint}'."
		)

	model = SRCNN().to(context.device)
	checkpoint_data = torch.load(checkpoint, map_location="cpu")
	state_dict = checkpoint_data.get("state_dict", checkpoint_data)
	key_map = {
		"features.0.weight": "conv1.weight",
		"features.0.bias": "conv1.bias",
		"map.0.weight": "conv2.weight",
		"map.0.bias": "conv2.bias",
		"reconstruction.weight": "conv3.weight",
		"reconstruction.bias": "conv3.bias",
	}
	cleaned = {}
	for key, value in state_dict.items():
		normalized_key = key[7:] if key.startswith("module.") else key
		cleaned[key_map.get(normalized_key, normalized_key)] = value
	model.load_state_dict(cleaned, strict=True)
	model.eval()
	return model


def infer_srcnn(model, lr: torch.Tensor, _edge_lr: torch.Tensor, hr: torch.Tensor, _scale: int) -> torch.Tensor:
	bicubic = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
	luma = rgb_to_y_channel(bicubic)
	return model(luma).clamp(0, 1)


def get_cuda_runtime_ms(run_fn: Callable[[], torch.Tensor], device: torch.device):
	if device.type == "cuda":
		start = torch.cuda.Event(enable_timing=True)
		end = torch.cuda.Event(enable_timing=True)
		start.record()
		output = run_fn()
		end.record()
		torch.cuda.synchronize(device)
		return output, start.elapsed_time(end)

	start = torch.perf_counter()
	output = run_fn()
	end = torch.perf_counter()
	return output, (end - start) * 1000.0


def default_specs(external_root: Path) -> list[ModelSpec]:
	return [
		ModelSpec("bicubic", None, load_builtin, infer_bicubic),
		ModelSpec("fsr", None, load_builtin, infer_fsr),
		ModelSpec("edge_guided_srcnn", Path("outputs/checkpoints/best.pth"), load_local_model, infer_local),
		ModelSpec(
			"srcnn_x4",
			external_root / "SRCNN-PyTorch/results/pretrained_models/srcnn_x4.pth.tar",
			load_srcnn_model,
			infer_srcnn,
		),
		ModelSpec("imdn_x4", external_root / "IMDN/checkpoints/IMDN_x4.pth", load_imdn_model, infer_single_input),
		ModelSpec("rfdn_x4", external_root / "RFDN/trained_model/RFDN_AIM.pth", load_rfdn_model, infer_single_input),
	]


def run_benchmark(args):
	device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
	external_root = Path(args.external_root)
	context = BenchmarkContext(scale=args.scale, device=device, external_root=external_root)
	dataset = DIV2K_Validation(hr_dir=args.val_hr, lr_dir=args.val_lr, scale=args.scale)
	specs = {spec.name: spec for spec in default_specs(external_root)}

	selected = []
	for model_name in args.models:
		if model_name not in specs:
			raise ValueError(f"Unknown model '{model_name}'. Available: {', '.join(specs)}")
		selected.append(specs[model_name])

	results = []
	for spec in selected:
		try:
			model = spec.loader(context, spec.checkpoint)
		except FileNotFoundError as exc:
			print(f"[Skip] {spec.name}: {exc}")
			continue

		ssim_sum = 0.0
		psnr_sum = 0.0
		runtime_sum = 0.0
		for idx in tqdm(range(len(dataset)), desc=f"Benchmark {spec.name}"):
			lr, edge_lr, hr, _edge_hr = dataset[idx]
			lr = lr.unsqueeze(0).to(device)
			edge_lr = edge_lr.unsqueeze(0).to(device)
			hr = hr.unsqueeze(0).to(device)

			with torch.no_grad():
				sr, runtime_ms = get_cuda_runtime_ms(
					lambda: spec.infer(model, lr, edge_lr, hr, args.scale),
					device,
				)

			target = hr
			if sr.size(1) != hr.size(1):
				if sr.size(1) == 1:
					target = rgb_to_y_channel(hr)
				else:
					raise ValueError(f"Unsupported channel mismatch for {spec.name}: SR has {sr.size(1)} channels, HR has {hr.size(1)}")

			ssim_sum += benchmark_ssim(sr, target, shave=args.scale, y_channel=sr.size(1) != 1).item()
			psnr_sum += benchmark_psnr(sr, target, shave=args.scale, y_channel=sr.size(1) != 1).item()
			runtime_sum += runtime_ms

		avg_ssim = ssim_sum / max(1, len(dataset))
		avg_psnr = psnr_sum / max(1, len(dataset))
		avg_runtime = runtime_sum / max(1, len(dataset))
		results.append(
			BenchmarkResult(
				name=spec.name,
				checkpoint=str(spec.checkpoint) if spec.checkpoint else "built-in",
				params=count_parameters(model),
				avg_y_ssim=avg_ssim,
				avg_y_psnr=avg_psnr,
				avg_runtime_ms=avg_runtime,
				samples=len(dataset),
			)
		)

	results.sort(key=lambda item: item.avg_y_ssim, reverse=True)
	print("\n=== Benchmark Results (Primary: Y-channel SSIM, Secondary: Y-channel PSNR) ===")
	print(f"Device: {device}")
	print(f"Dataset: {args.val_lr} -> {args.val_hr}")
	print(f"Scale: x{args.scale} | Shave: {args.scale}")
	print()
	print(f"{'Model':<20} {'SSIM-Y':>10} {'PSNR-Y':>10} {'Runtime(ms)':>14} {'Params':>14}")
	for item in results:
		print(f"{item.name:<20} {item.avg_y_ssim:>10.4f} {item.avg_y_psnr:>10.4f} {item.avg_runtime_ms:>14.4f} {item.params:>14}")

	if args.save_json:
		output_path = Path(args.save_json)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_text(json.dumps([asdict(item) for item in results], indent=2), encoding="utf-8")
		print(f"\nSaved JSON report to: {output_path}")


def main():
	parser = argparse.ArgumentParser(description="Benchmark lightweight SR models on DIV2K validation")
	parser.add_argument("--scale", type=int, default=4)
	parser.add_argument("--val_hr", type=str, default="./DIV2K/DIV2K_valid_HR")
	parser.add_argument("--val_lr", type=str, default="./DIV2K/DIV2K_valid_LR_bicubic/X4")
	parser.add_argument("--external_root", type=str, default="./external_benchmarks")
	parser.add_argument(
		"--models",
		nargs="+",
		default=["bicubic", "fsr", "edge_guided_srcnn", "srcnn_x4", "imdn_x4", "rfdn_x4"],
		help="Models to benchmark",
	)
	parser.add_argument("--save_json", type=str, default="./outputs/benchmark_results.json")
	parser.add_argument("--cpu", action="store_true")
	args = parser.parse_args()

	if args.val_lr.endswith("/") or args.val_lr.endswith("\\"):
		args.val_lr = f"{args.val_lr}X{args.scale}"

	run_benchmark(args)


if __name__ == "__main__":
	main()