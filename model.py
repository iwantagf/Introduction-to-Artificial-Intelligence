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
    def __init__(self, input_channels, edge_channels, internal_channels=64):
        super().__init__()
        # SRCNN-inspired backbone with Edge Injection at every layer
        # Layer 1
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=9, padding=4)
        self.act1 = nn.SiLU(inplace=True)
        
        # Layer 2: Injection. Input = 64 (from L1) + edge_channels
        self.conv2 = nn.Conv2d(64 + edge_channels, 32, kernel_size=5, padding=2)
        self.act2 = nn.SiLU(inplace=True)
        
        # Layer 3: Injection. Input = 32 (from L2) + edge_channels
        self.conv3 = nn.Conv2d(32 + edge_channels, input_channels, kernel_size=5, padding=2)

    def forward(self, x, edge_feat):
        # Layer 1
        x = self.act1(self.conv1(x))
        
        # Injection 1: Concatenate edge features
        x = torch.cat([x, edge_feat], dim=1)
        x = self.act2(self.conv2(x))
        
        # Injection 2: Concatenate edge features
        x = torch.cat([x, edge_feat], dim=1)
        x = self.conv3(x)
        
        return x

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
            # Edge head taking 2 channels (Gx, Gy)
            # We want this to produce a feature map useful for injection.
            # Let's say we project it to 'head_features' size initially for fusion,
            # but also keep a version for injection (or reuse the same).
            # To keep dims consistent for SRCNN backbone injection, we'll define dim.
            self.edge_feat_dim = 32
            self.edge_head = nn.Sequential(
                nn.Conv2d(2, self.edge_feat_dim, kernel_size=9, padding=4),
                nn.SiLU(inplace=True),
            )
            
            # Fusion: RGB (head_features) + Edge (edge_feat_dim) -> fused -> num_features
            self.fusion = nn.Sequential(
                nn.Conv2d(head_features + self.edge_feat_dim, num_features, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
            )
        else:
            self.head = nn.Sequential(
                nn.Conv2d(input_channels, head_features, kernel_size=9, padding=4),
                nn.SiLU(inplace=True),
            )
            self.fusion = nn.Conv2d(head_features, num_features, kernel_size=3, padding=1)

        # SRCNN Backbone now takes Edge Injection
        # We pass 'edge_feat_dim' as the size of extra channels
        edge_ch = self.edge_feat_dim if use_edge_branch else 0
        self.body = SRCNNBackbone(num_features, edge_channels=edge_ch)

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
            # edge input is [B, 2, H, W]
            edge_feat = self.edge_head(edge) # -> [B, 32, H, W]
            
            # Fusion at input
            x = torch.cat([rgb_feat, edge_feat], dim=1)
            x = self.fusion(x)
            
            # Pass edge features to backbone for deep injection
            x = self.body(x, edge_feat)
        else:
            x = self.head(lr)
            x = self.fusion(x)
            x = self.body(x, None)
        
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