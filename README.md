# Edge-Guided SRCNN (FSR-Inspired)

This project implements a Single Image Super-Resolution (SISR) model that combines the classic **SRCNN** architecture with **Edge-Guided Learning** inspired by AMD's FSR (FidelityFX Super Resolution) 1.0 technology. It is designed to be a lightweight, efficient upscaler that focuses on reconstructing sharp edges while suppressing noise.

## Key Features

*   **Residual SRCNN Backbone:** Uses 3 residual edge-guided blocks (6 convolutions total) to preserve the lightweight SRCNN spirit while improving gradient flow.
*   **SiLU Activation:** Uses **SiLU (Swish)** instead of ReLU for smoother gradient flow and better performance in deep networks.
*   **Edge Attention Mechanism:** Instead of simple concatenation, the model uses an explicit attention gate (`RGB_Feat * Sigmoid(Edge_Feat)`) to boost features in high-frequency regions.
*   **Deep Edge Gating:** Uses edge-conditioned gates inside every backbone stage, replacing direct concatenation with multiplicative feature modulation.
*   **Directional Edge Detection:** Uses Sobel gradients ($G_x, G_y$) instead of simple intensity differences, providing the network with directional context for edge reconstruction.
    *   **Preprocessing:** Applies thresholding and conditional blur based on gradient magnitude.
*   **Edge-Aware Loss:** Training uses Charbonnier loss for RGB reconstruction plus an MSE edge-consistency loss.
*   **Global Residual Learning:** The bicubic-upsampled LR image is added inside the model forward path, so the network only learns the missing high-frequency residual.
*   **Scale-Aware Edge Preprocessing:** Edge maps are thresholded and blurred with blur strength adjusted to the target scale.
*   **Evaluation Before RCAS:** PSNR is measured on the raw model output; RCAS is applied only for saved previews.
*   **Benchmark PSNR:** Validation now reports benchmark-style PSNR on the Y channel with border shaving equal to the upscaling factor, alongside raw RGB PSNR for internal comparison.

## Algorithm Description

### 1. Edge Extraction (Directional)
We use Sobel filters to compute gradients in X and Y directions, capturing directional edge information.
$$ Edge = [\tanh(G_x), \tanh(G_y)] $$
The output is a 2-channel tensor with values approximately in $(-1, 1)$.

### 2. Network Architecture
The model takes two inputs: the Low-Resolution (LR) RGB image and its corresponding 2-channel Edge Map.

1.  **Dual Heads:**
    *   `RGB Head`: 3 -> 96 channels (9x9 Conv) + **SiLU**.
    *   `Edge Head`: 2 -> 32 channels (9x9 Conv) + **SiLU**.
2.  **Edge Attention:**
    *   `Feat = RGB_Head(LR)`
    *   `Edge_Weights = Sigmoid(Edge_Head(Edge))`
    *   `Weighted_Feat = Feat * Edge_Weights`
3.  **Backbone (Residual Edge-Guided SRCNN):**
    The fused feature map is processed by 3 residual blocks. Each block contains 2 convolutions and uses edge gating before each convolution.
    *   `Gate = Sigmoid(Conv(Edge_Enc))`
    *   `Feat = Feat * Gate + Feat`
    *   `BlockOut = BlockIn + Conv(SiLU(Conv(Feat)))`
4.  **Upsampling:** Sub-pixel convolution (PixelShuffle). For `scale=4`, the model uses two `x2` stages instead of one direct `x4` stage.
5.  **Global Residual:** Final output = `Residual_Branch(LR, Edge) + Bicubic(LR)`.

### 3. Training Strategy
*   **Loss:** $L_{total} = Charbonnier(SR, HR) + \lambda \cdot MSE(Edge(SR), Edge(HR))$
*   **Lambda ($\lambda$):** 0.015
*   **Optimizer:** Adam (`lr=2e-4`).
*   **Scheduler:** CosineAnnealingLR (with 5-epoch Linear Warmup).

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

## Lightweight Comparison on DIV2K

The table below mixes two kinds of evidence:

*   **Local validation** for this repository using `python valid.py --scale 4 --lr_mode bicubic --checkpoint outputs/checkpoints/best.pth`.
*   **Official paper / challenge / model-zoo references** for other lightweight or compact baselines.

Because these sources do **not** all use the same protocol, the numbers should be read as a practical orientation rather than a strict leaderboard. The main differences are:

*   Some sources evaluate on **Y channel**, while others evaluate on **RGB**.
*   Some use **DIV2K validation 0801-0900 / DIV2K100**, while challenge papers report **DIV2K-based challenge tracks**.
*   Border cropping is not always identical.

| Model | Paper / Reference | Approx. Params | DIV2K-Related Result | Evaluation Protocol |
| :--- | :--- | :---: | :--- | :--- |
| Edge-Guided SRCNN (this repo) | Local result from this repository | Default repo config | **29.86 dB** on DIV2K validation bicubic x4 | Y channel, `shave=4`, measured with the current `valid.py` |
| [EDSR-baseline x4](https://arxiv.org/abs/1707.02921) | Official [EDSR PyTorch implementation](https://github.com/sanghyun-son/EDSR-PyTorch) and [BasicSR Model Zoo](https://github.com/XPixelGroup/BasicSR/blob/master/docs/ModelZoo.md) | **1.52M** | **28.95 - 28.97 dB** on DIV2K100 / DIV2K 0801-0900 | RGB evaluation, border crop by scale or `scale + 2` depending on the official code path |
| [IMDN](https://arxiv.org/abs/1909.11856) | [AIM 2019 Constrained SR report](https://arxiv.org/abs/1911.01249) | Not stated in accessible DIV2K challenge text here | **1st place** in AIM 2019 constrained x4 SR tracks built on DIV2K | Challenge-oriented comparison; paper/repo focus more on challenge ranking and standard benchmarks than a single plain-text DIV2K100 x4 score |
| [RFDN](https://arxiv.org/abs/2009.11551) | [AIM 2020 Efficient SR report](https://arxiv.org/abs/2009.06943) | Not stated in accessible DIV2K challenge text here | **1st place** in AIM 2020 efficient x4 SR challenge on DIV2K-based data | Challenge-oriented comparison emphasizing efficiency, runtime, and fidelity trade-offs |
| Compact residual baseline (MSRResNet x4) | [SRGAN / SRResNet](https://arxiv.org/abs/1609.04802) via [BasicSR Model Zoo](https://github.com/XPixelGroup/BasicSR/blob/master/docs/ModelZoo.md) | Not listed in the fetched model-zoo snippet | **28.9967 dB** on DIV2K100 | RGB evaluation with crop equal to scale in BasicSR |

### Reading This Table Correctly

*   The **current model** is strongest when you care about a small, edge-focused model and want a reproducible local DIV2K validation result from this repo.
*   **EDSR-baseline** is included as a compact reference point, even though it is not the most lightweight model in the table.
*   **IMDN** and **RFDN** are important lightweight references because they are strong efficient-SR families that performed well in DIV2K-based AIM challenges.
*   If you want a perfectly apples-to-apples comparison, all models should be re-evaluated under one shared script on the same split and the same metric definition, for example Y-channel PSNR with `shave=4` at x4.
