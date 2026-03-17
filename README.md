# Edge-Guided SRCNN (FSR-Inspired)

This project implements a Single Image Super-Resolution (SISR) model that combines the classic **SRCNN** architecture with **Edge-Guided Learning** inspired by AMD's FSR (FidelityFX Super Resolution) 1.0 technology. It is designed to be a lightweight, efficient upscaler that focuses on reconstructing sharp edges while suppressing noise.

## Key Features

*   **Modified SRCNN Backbone:** Replaces typical heavy Residual Blocks with the classic 3-layer SRCNN structure (9-5-5 kernels), adapted for feature extraction. This significantly reduces model complexity.
*   **SiLU Activation:** Uses **SiLU (Swish)** instead of ReLU for smoother gradient flow and better performance in deep networks.
*   **Edge Attention Mechanism:** Instead of simple concatenation, the model uses an explicit attention gate (`RGB_Feat * Sigmoid(Edge_Feat)`) to boost features in high-frequency regions.
*   **Deep Edge Injection:** Concatenates directional edge gradients into *every* layer of the SRCNN backbone, ensuring the network can guide its feature extraction with explicit edge information at all depths.
*   **Directional Edge Detection:** Uses Sobel gradients ($G_x, G_y$) instead of simple intensity differences, providing the network with directional context for edge reconstruction.
    *   **Preprocessing:** Applies thresholding and conditional blur based on gradient magnitude.
*   **Edge-Aware Loss:** Training is guided by a composite loss function that minimizes pixel-wise error (MSE) while enforcing edge consistency (Edge Loss) with a weight of `0.015`.
*   **Global Residual Learning:** The model learns the *residual* difference between the target HR image and the bicubic-upsampled LR image, ensuring detailed texture recovery.

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
3.  **Backbone (Edge-Injected SRCNN):**
    Each convolutional layer in the SRCNN backbone receives the encoded edge features via concatenation.
    *   `Input`: Concat(Weighted_Feat, Edge_Enc) -> Conv 9x9 (64 filters) + **SiLU**.
    *   `Hidden`: Concat(Feat, Edge_Enc) -> Conv 5x5 (32 filters) + **SiLU**.
    *   `Output`: Concat(Feat, Edge_Enc) -> Conv 5x5 (64 filters).
4.  **Upsampling:** Sub-pixel convolution (PixelShuffle) to reach target resolution.
5.  **Global Residual:** Final output = `Model(LR) + Bicubic(LR)`.

### 3. Training Strategy
*   **Loss:** $L_{total} = MSE(SR, HR) + \lambda \cdot MSE(Edge(SR), Edge(HR))$
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
