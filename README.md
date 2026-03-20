# Edge-Guided SRCNN (FSR-Inspired)

This project implements a Single Image Super-Resolution (SISR) model that combines the classic **SRCNN** architecture with **Edge-Guided Learning** inspired by AMD's FSR (FidelityFX Super Resolution) 1.0 technology. It is designed to be a lightweight, efficient upscaler that focuses on reconstructing sharp edges while suppressing noise.

## Key Features

*   **Residual SRCNN Backbone:** Uses 3 residual edge-guided blocks (6 convolutions total) to preserve the lightweight SRCNN spirit while improving gradient flow.
*   **SiLU Activation:** Uses **SiLU (Swish)** instead of ReLU for smoother gradient flow and better performance in deep networks.
*   **RGB-Edge Feature Fusion:** RGB and edge features are first encoded by separate heads, then fused by concatenation followed by a `3x3` convolution before entering the backbone.
*   **Spatial Attention Refinement:** Uses channel-pooled spatial attention after feature fusion and inside each residual block so the network can emphasize informative regions and textured structures.
*   **Deep Edge Gating:** Uses edge-conditioned gates inside every backbone stage, replacing direct concatenation with multiplicative feature modulation.
*   **Directional Edge Detection:** Uses Sobel gradients ($G_x, G_y$) instead of simple intensity differences, providing the network with directional context for edge reconstruction.
    *   **Preprocessing:** Applies thresholding and conditional blur based on gradient magnitude.
*   **Edge-Aware Loss:** Training uses MSE reconstruction loss plus an MSE edge-consistency loss.
*   **Global Residual Learning:** The network predicts a residual image, and the bicubic-upsampled LR image is added outside the backbone to recover the final SR output.
*   **Scale-Aware Edge Preprocessing:** Edge maps are thresholded and blurred with blur strength adjusted to the target scale.
*   **Evaluation Before RCAS:** PSNR is measured on the raw model output; RCAS is applied only for saved previews.
*   **Benchmark Metrics:** Validation reports **SSIM on the Y channel** as the primary benchmark metric and **PSNR on the Y channel** as the secondary metric, both with border shaving equal to the upscaling factor.

## Algorithm Description

### 1. Edge Extraction (Directional)
We use Sobel filters to compute gradients in X and Y directions, capturing directional edge information.
$$ Edge = [\tanh(G_x), \tanh(G_y)] $$
The output is a 2-channel tensor with values approximately in $(-1, 1)$.

### 2. Network Architecture
The model takes two inputs: the Low-Resolution (LR) RGB image and its corresponding 2-channel Edge Map.

1.  **Dual Heads:**
    *   `RGB Head`: 3 -> 48 channels (9x9 Conv) + **SiLU**.
    *   `Edge Head`: 2 -> 32 channels (9x9 Conv) + **SiLU**.
2.  **Feature Fusion:**
    *   `RGB_Feat = RGB_Head(LR)`
    *   `Edge_Feat = Edge_Head(Edge)`
    *   `Fused_Feat = Conv([RGB_Feat, Edge_Feat])`
    *   `Attended_Feat = SpatialAttention(Fused_Feat)`
3.  **Backbone (Residual Edge-Guided SRCNN):**
    The fused 64-channel feature map is processed by 3 residual edge-guided blocks. Each block contains 2 convolutions, uses edge gating before each convolution, and applies spatial attention before the residual merge.
    *   `Gate = Sigmoid(Conv(Edge_Enc))`
    *   `Feat = Feat * Gate + Feat`
    *   `Feat = SpatialAttention(Conv(SiLU(Conv(Feat))))`
    *   `BlockOut = BlockIn + 0.3 * Feat`
4.  **Upsampling:** Sub-pixel convolution (PixelShuffle). For `scale=4`, the model uses two `x2` stages instead of one direct `x4` stage.
5.  **Residual Reconstruction:** The network predicts an HR residual image.
6.  **Global Residual:** Final SR result used for loss and evaluation is `Residual_Branch(LR, Edge) + Bicubic(LR)`.

### 3. Training Strategy
*   **Loss:** $L_{total} = MSE(SR, HR) + \lambda \cdot MSE(Edge(SR), Edge(HR))$
*   **Lambda ($\lambda$):** 0.015
*   **Optimizer:** Adam (`lr=2e-4`).
*   **Scheduler:** CosineAnnealingWarmRestarts.
*   **EMA:** Exponential Moving Average with decay `0.999` is maintained during training.
*   **Edge-Loss Schedule:** `lambda_edge` is increased, held, then decayed so late training focuses more on pixel fidelity.

### 4. Evaluation Protocol
*   **Primary metric:** SSIM on the **Y channel only**.
*   **Secondary metric:** PSNR on the **Y channel only**.
*   **Border crop:** `shave = scale`, so x4 evaluation removes 4 pixels on each border.
*   **Preview sharpening:** RCAS is preview-only and is **not** used when measuring PSNR.
*   **Checkpoint selection:** `best.pth` is selected by the primary Y-channel benchmark metric.

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

To validate `best.pth` on the classic benchmark suite (`Set5`, `Set14`, `Urban100`, `Manga109`, `B100`):

```bash
python valid.py --checkpoint outputs/checkpoints/best.pth --scale 4 --run_benchmark_suite --benchmark_root ./benchmarks
```

Expected benchmark layout:

```text
benchmarks/
    Set5/
        HR/
        LR_bicubic/
            X4/
    Set14/
        HR/
        LR_bicubic/
            X4/
    Urban100/
        HR/
        LR_bicubic/
            X4/
    Manga109/
        HR/
        LR_bicubic/
            X4/
    B100/
        HR/
        LR_bicubic/
            X4/
```

### 4. Results
During training, preview images are saved in `outputs/preview/`. The model automatically saves the best checkpoint to `outputs/checkpoints/best.pth`.

## Model Performance
Due to its lightweight nature, this model focuses on **efficiency** and **edge crispness** rather than achieving state-of-the-art PSNR scores. It is suitable for real-time applications or mobile deployment.

## Classic Benchmark Suite Results

The benchmark suite in this repository currently reports **SSIM on the Y channel** as the primary metric and **PSNR on the Y channel** as the secondary metric.

All numbers below follow the same local protocol:

*   Scale: x4
*   Border crop: `shave=4`
*   No RCAS or other post-processing during measurement
*   Datasets run locally: `Set5`, `Set14`, `Urban100`, `B100`
*   `Manga109` was skipped in the local run because the dataset was not available in the workspace

### Edge-Guided SRCNN Local Results

| Dataset | SSIM(Y) | PSNR(Y) | Bicubic SSIM(Y) | Bicubic PSNR(Y) | Samples |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Set5 | **0.8877** | **31.43** | 0.8208 | 28.63 | 5 |
| Set14 | **0.7766** | **28.14** | 0.7178 | 26.21 | 14 |
| Urban100 | **0.7581** | **25.26** | 0.6664 | 23.24 | 100 |
| B100 | **0.7342** | **27.29** | 0.6831 | 26.04 | 100 |

These local results show consistent gains over bicubic across every evaluated benchmark set.

### Detailed Local Benchmark Summary

| Dataset | Loss | Min SSIM(Y) | Max SSIM(Y) | Min PSNR(Y) | Max PSNR(Y) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Set5 | 0.0021 | 0.8023 | 0.9309 | 27.15 | 33.56 |
| Set14 | 0.0049 | 0.5291 | 0.9358 | 22.89 | 33.69 |
| Urban100 | 0.0084 | 0.4070 | 0.9602 | 17.39 | 35.45 |
| B100 | 0.0056 | 0.3373 | 0.9801 | 19.82 | 39.27 |

### Classical SR Reference Table

The table below adds several classical reference models shown in the comparison image provided for this project. Values are presented in `PSNR / SSIM` format.

| Model (Depth) | Set5 | Set14 | B100 |
| :--- | :---: | :---: | :---: |
| Bicubic (Baseline) | 28.42 / 0.8104 | 26.00 / 0.7027 | 25.96 / 0.6675 |
| SRCNN (3 layers) | 30.48 / 0.8628 | 27.50 / 0.7513 | 26.90 / 0.7101 |
| FSRCNN (8 layers) | 30.72 / 0.8657 | 27.61 / 0.7550 | 26.98 / 0.7150 |
| VDSR (20 layers) | 31.35 / 0.8838 | 28.01 / 0.7674 | 27.29 / 0.7251 |
| Edge-Guided SRCNN (this repo, local x4 run) | **31.43 / 0.8877** | **28.14 / 0.7766** | **27.29 / 0.7342** |

Under this local benchmark run, the current model is competitive with the classical references above and exceeds the listed Bicubic, SRCNN, FSRCNN, and VDSR numbers on Set5 and Set14, while matching VDSR's PSNR on B100 and improving its SSIM.
