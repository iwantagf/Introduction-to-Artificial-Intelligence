# Edge-Guided SRCNN (FSR-Inspired)

This project implements a Single Image Super-Resolution (SISR) model that combines the classic **SRCNN** architecture with **Edge-Guided Learning** inspired by AMD's FSR (FidelityFX Super Resolution) 1.0 technology. It is designed to be a lightweight, efficient upscaler that focuses on reconstructing sharp edges while suppressing noise.

## Key Features

*   **Modified SRCNN Backbone:** Replaces typical heavy Residual Blocks with the classic 3-layer SRCNN structure (9-5-5 kernels), adapted for feature extraction. This significantly reduces model complexity.
*   **SiLU Activation:** Uses **SiLU (Swish)** instead of ReLU for smoother gradient flow and better performance in deep networks.
*   **Edge Attention Mechanism:** Instead of simple concatenation, the model uses an explicit attention gate (`RGB_Feat * Sigmoid(Edge_Feat)`) to boost features in high-frequency regions.
*   **FSR-Style Edge Detection:** Implements an edge detection algorithm based on AMD FSR's EASU (Edge Adaptive Spatial Upsampling). It calculates edge strength using luminance difference analysis on a cross-pattern grid, normalized to [0, 1].
*   **Edge Preprocessing Pipeline:** Features a robust preprocessing step for edge maps:
    *   **Thresholding:** Removes weak noise (values < 0.1).
    *   **Conditional Gaussian Blur:** Automatically applies smoothing if the edge map is detected as too noisy (mean intensity > 0.15).
*   **Edge-Aware Loss:** Training is guided by a composite loss function that minimizes pixel-wise error (MSE) while enforcing edge consistency (Edge Loss) with a weight of `0.015`.
*   **Global Residual Learning:** The model learns the *residual* difference between the target HR image and the bicubic-upsampled LR image, ensuring detailed texture recovery.

## Algorithm Description

### 1. Edge Extraction (FSR-EASU)
Instead of standard Sobel filters, we use an approximation of the "len" (length/strength) parameter from FSR 1.0. 
For a center pixel $f$ and its neighbors $b, h, e, g$ (top, bottom, left, right):
$$ Edge = |b - f| + |h - f| + |e - f| + |g - f| $$
This map is then clamped to the range $[0, 1]$.

### 2. Network Architecture
The model takes two inputs: the Low-Resolution (LR) RGB image and its corresponding Edge Map.

1.  **Dual Heads:**
    *   `RGB Head`: 3 -> 96 channels (9x9 Conv) + **SiLU**.
    *   `Edge Head`: 1 -> 96 channels (9x9 Conv) + **SiLU**.
2.  **Edge Attention:**
    *   `Feat = RGB_Head(LR) * Sigmoid(Edge_Head(Edge))`
    *   This acts as a spatial gate, amplifying features where edge probability is high.
3.  **Backbone (SRCNN):**
    *   Layer 1: Conv 9x9 (64 filters) + **SiLU**.
    *   Layer 2: Conv 5x5 (32 filters) + **SiLU**.
    *   Layer 3: Conv 5x5 (64 filters).
4.  **Upsampling:** Sub-pixel convolution (PixelShuffle) to reach target resolution.
5.  **Global Residual:** Final output = `Model(LR) + Bicubic(LR)`.

### 3. Training Strategy
*   **Loss:** $L_{total} = MSE(SR, HR) + \lambda \cdot MSE(Edge(SR), Edge(HR))$
*   **Lambda ($\lambda$):** 0.015
*   **Optimizer:** Adam (`lr=2e-4`).
*   **Scheduler:** ReduceLROnPlateau.

## Project Structure

```
Introduction-to-Artificial-Intelligence/
├── dataset.py          # FSR edge extraction, DIV2K loading
├── model.py            # EdgeGuidedCNN with SRCNNBackbone
├── train.py            # Training loop with Edge Preprocessing
├── valid.py            # Validation script
├── main.py             # Entry point
├── fsr_benchmark.py    # Standalone FSR edge-guided benchmark
├── README.md           # Documentation
└── outputs/            # Checkpoints and previews
```

## Requirements

*   Python 3.8+
*   PyTorch 1.7+
*   Torchvision
*   Pillow
*   Tqdm

## Usage

### 1. Prepare Dataset (DIV2K)
Download the DIV2K dataset and place it in the `DIV2K` folder with the following structure:
```
DIV2K/
  DIV2K_train_HR/
  DIV2K_train_LR_bicubic/
    X4/
  DIV2K_valid_HR/
  DIV2K_valid_LR_bicubic/
    X4/
```

### 2. Training
To train the model for 4x upscaling using the mixed dataset mode (Bicubic + Unknown kernels):

```bash
python main.py --scale 4 --epochs 100 --batch_size 16 --lambda_edge 0.015
```

**Common Arguments:**
*   `--scale`: Upscaling factor (default: 4).
*   `--lr`: Learning rate (default: 2e-4).
*   `--lambda_edge`: Edge loss weight (default: 0.015).

### 3. Validation / Inference
To test a trained checkpoint:

```bash
python valid.py --checkpoint outputs/checkpoints/best.pth --scale 4
```

### 4. Results
During training, preview images are saved in `outputs/preview/`. The model automatically saves the best checkpoint (highest PSNR) to `outputs/checkpoints/best.pth`.

## Model Performance
Due to its lightweight nature, this model focuses on **efficiency** and **edge crispness** rather than achieving state-of-the-art PSNR scores. It is suitable for real-time applications or mobile deployment.
