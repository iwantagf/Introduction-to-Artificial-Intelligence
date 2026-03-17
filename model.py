import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rcas(img: torch.Tensor, strength: float = 0.5) -> torch.Tensor:
    # Simple edge detector kernel for sharpening
    kernel = torch.tensor([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=img.dtype, device=img.device).view(1, 1, 3, 3) / 5.0
    
    # Apply sharpening per channel
    b, c, h, w = img.shape
    sharpened = img.clone()
    
    for i in range(c):
        channel = img[:, i:i+1, :, :]
        sharp = F.conv2d(channel, kernel, padding=1)
        # Blend original and sharpened
        sharpened[:, i:i+1, :, :] = img[:, i:i+1, :, :] + strength * (sharp - channel)
    
    return sharpened.clamp(0, 1)


class SRCNNBackbone(nn.Module):
    def __init__(self, input_channels, internal_channels=64):
        super().__init__()
        # SRCNN-inspired backbone: 9-5-5 structure
        # Adapted to preserve spatial dimensions (padding) and input/output channel count
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=9, padding=4),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2), # Original SRCNN used 1x1 or 5x5 here
            nn.SiLU(inplace=True),
            nn.Conv2d(32, input_channels, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.net(x)

class EdgeGuidedCNN(nn.Module):
    def __init__(self, input_channels: int = 4, num_features: int = 64, head_features: int = 96, scale: int = 2, num_blocks: int = 8, use_edge_branch: bool = True):
        super().__init__()
        
        self.use_edge_branch = use_edge_branch
        
        if use_edge_branch:
            # Separate branches for RGB and edge with balanced capacity
            self.rgb_head = nn.Sequential(
                nn.Conv2d(3, head_features, kernel_size=9, padding=4),
                nn.SiLU(inplace=True),
            )
            # Edge head must match rgb_head channels for element-wise attention
            self.edge_head = nn.Sequential(
                nn.Conv2d(1, head_features, kernel_size=9, padding=4),
                nn.SiLU(inplace=True),
            )
            # Fusion: No concatenation anymore, just projection after attention
            # Input dim is head_features due to element-wise multiplication
            self.fusion = nn.Sequential(
                nn.Conv2d(head_features, num_features, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
            )
        else:
            # Original unified head
            self.head = nn.Sequential(
                nn.Conv2d(input_channels, head_features, kernel_size=9, padding=4),
                nn.SiLU(inplace=True),
            )
            self.fusion = nn.Conv2d(head_features, num_features, kernel_size=3, padding=1)

        # Replaced ResBlock backbone with SRCNN structure
        self.body = SRCNNBackbone(num_features)

        self.upsampler = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale), # Upscale
            nn.SiLU(inplace=True),
        )

        self.tail = nn.Conv2d(num_features, 3, kernel_size=3, padding=1)

        self._init_weights()

    def forward(self, lr: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        if self.use_edge_branch:
            rgb_feat = self.rgb_head(lr)
            edge_feat = self.edge_head(edge)
            
            # Edge Attention: Modulate RGB features with edge probability
            # "Boost high-frequency" logic implies we want to emphasize features where edges are present
            x = rgb_feat * torch.sigmoid(edge_feat)
            
            x = self.fusion(x)
        else:
            x = self.head(lr)
            x = self.fusion(x)
        
        x = self.body(x)
        x = self.upsampler(x)
        sr = self.tail(x)
        return sr

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize tail layer with near-zero weights so the model starts
        # by predicting a near-zero residual (output equivalent to bicubic upsampling)
        nn.init.normal_(self.tail.weight, mean=0, std=0.001)
        if self.tail.bias is not None:
            nn.init.zeros_(self.tail.bias)