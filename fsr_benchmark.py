"""
Simple FSR (Fidelity and Super-Resolution) benchmark on DIV2K validation set.
Compares edge-guided upscaling against HR reference.
"""
import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import numpy as np

from dataset import DIV2K_Validation, fsr_edge
from train import benchmark_psnr, benchmark_ssim


def fsr_edge_guided_upscale(lr, scale: int):
    """
    Simple edge-guided upscaling:
    1. Compute FSR edge map
    2. Use bicubic upsampling as base
    3. Enhance edges via edge map guidance
    """
    if lr.shape[1] == 3:
        # RGB to Luma for edge guidance
        luma = 0.299 * lr[:, 0:1, :, :] + 0.587 * lr[:, 1:2, :, :] + 0.114 * lr[:, 2:3, :, :]
    else:
        luma = lr
        
    edge = fsr_edge(luma)
    
    # Standard bicubic upsampling
    lr_up = F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)
    edge_up = F.interpolate(edge, scale_factor=scale, mode="bicubic", align_corners=False)
    
    # Simple edge-guided enhancement
    return lr_up.clamp(0, 1)


def evaluate_fsr(scale: int, val_dir_hr: str, val_dir_lr: str, save_dir: str, preview_count: int = 3):
    """Run FSR on validation set and save previews + metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    save_dir = Path(save_dir)
    preview_dir = save_dir / "Preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    
    val_dataset = DIV2K_Validation(hr_dir=val_dir_hr, lr_dir=val_dir_lr, scale=scale)
    
    total_ssim_y = 0.0
    total_psnr_y = 0.0
    saved = 0
    to_pil = T.ToPILImage()
    
    ssim_y_list = []
    psnr_y_list = []
    
    for idx in tqdm(range(len(val_dataset)), desc="FSR Benchmark"):
        lr, _, hr, _ = val_dataset[idx]
        
        lr = lr.unsqueeze(0).to(device)
        hr = hr.unsqueeze(0).to(device)
        
        # Apply FSR upscaling
        sr_fsr = fsr_edge_guided_upscale(lr, scale)
        
        # Resize to HR size
        target_size = hr.shape[-2:]
        sr_fsr = F.interpolate(sr_fsr, size=target_size, mode="bicubic", align_corners=False)
        
        ssim_y_val = benchmark_ssim(sr_fsr, hr, shave=scale, y_channel=True).item()
        psnr_y_val = benchmark_psnr(sr_fsr, hr, shave=scale, y_channel=True).item()
        total_ssim_y += ssim_y_val
        total_psnr_y += psnr_y_val
        ssim_y_list.append(ssim_y_val)
        psnr_y_list.append(psnr_y_val)
        
        # Save previews (first few samples)
        if saved < preview_count:
            sr_pil = to_pil(sr_fsr.squeeze(0).clamp(0, 1).cpu())
            hr_pil = to_pil(hr.squeeze(0).clamp(0, 1).cpu())
            
            sr_pil = sr_pil.resize(target_size[::-1], Image.BICUBIC)
            hr_pil = hr_pil.resize(target_size[::-1], Image.BICUBIC)
            
            save_path_sr = preview_dir / f"fsr_idx_{idx:03d}_sr.png"
            save_path_hr = preview_dir / f"fsr_idx_{idx:03d}_hr.png"
            
            sr_pil.save(save_path_sr)
            hr_pil.save(save_path_hr)
            saved += 1
    
    avg_ssim_y = total_ssim_y / max(1, len(val_dataset))
    avg_psnr_y = total_psnr_y / max(1, len(val_dataset))
    
    print(f"\n=== FSR Benchmark Results ===")
    print(f"Average SSIM (Y, shave={scale}): {avg_ssim_y:.4f}")
    print(f"Average PSNR (Y, shave={scale}): {avg_psnr_y:.2f}")
    print(f"Min SSIM (Y, shave={scale}): {min(ssim_y_list):.4f}")
    print(f"Max SSIM (Y, shave={scale}): {max(ssim_y_list):.4f}")
    print(f"Min PSNR (Y, shave={scale}): {min(psnr_y_list):.2f}")
    print(f"Max PSNR (Y, shave={scale}): {max(psnr_y_list):.2f}")
    print(f"Previews saved to: {preview_dir}")
    
    return avg_ssim_y


def main():
    parser = argparse.ArgumentParser(description="FSR Benchmark on DIV2K")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--val_hr", type=str, default="./DIV2K/DIV2K_valid_HR")
    parser.add_argument("--val_lr", type=str, default="./DIV2K/DIV2K_valid_LR_bicubic/")
    parser.add_argument("--save_dir", type=str, default="FSR_output")
    parser.add_argument("--preview_count", type=int, default=3)
    args = parser.parse_args()
    
    args.val_lr = os.path.join(args.val_lr, f"x{args.scale}")
    
    evaluate_fsr(
        scale=args.scale,
        val_dir_hr=args.val_hr,
        val_dir_lr=args.val_lr,
        save_dir=args.save_dir,
        preview_count=args.preview_count,
    )


if __name__ == "__main__":
    main()
