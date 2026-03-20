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


class EdgeGate(nn.Module):
    def __init__(self, in_channels, edge_channels):
        super().__init__()
        self.gate_conv = nn.Conv2d(edge_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, edge):
        gate = torch.sigmoid(self.gate_conv(edge))
        return x * gate + x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.attn = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.sigmoid(self.attn(torch.cat([avg_map, max_map], dim=1)))
        return x * attention + x


class ResidualEdgeBlock(nn.Module):
    def __init__(self, channels, edge_channels, kernel_size1=5, kernel_size2=5, residual_scale=0.1):
        super().__init__()
        self.residual_scale = residual_scale
        self.gate1 = EdgeGate(channels, edge_channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size1, padding=kernel_size1 // 2)
        self.act1 = nn.SiLU(inplace=True)
        self.gate2 = EdgeGate(channels, edge_channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size2, padding=kernel_size2 // 2)
        self.spatial_attention = SpatialAttention()

    def forward(self, x, edge_feat):
        residual = x

        if edge_feat is not None:
            x = self.gate1(x, edge_feat)
        x = self.act1(self.conv1(x))

        if edge_feat is not None:
            x = self.gate2(x, edge_feat)
        x = self.conv2(x)
        x = self.spatial_attention(x)

        return residual + self.residual_scale * x


class SRCNNBackbone(nn.Module):
    def __init__(self, input_channels, edge_channels, internal_channels=64):
        super().__init__()
        self.block1 = ResidualEdgeBlock(input_channels, edge_channels, kernel_size1=9, kernel_size2=5, residual_scale=0.3)
        self.block2 = ResidualEdgeBlock(input_channels, edge_channels, kernel_size1=5, kernel_size2=5, residual_scale=0.3)
        self.block3 = ResidualEdgeBlock(input_channels, edge_channels, kernel_size1=5, kernel_size2=5, residual_scale=0.3)

    def forward(self, x, edge_feat):
        x = self.block1(x, edge_feat)
        x = self.block2(x, edge_feat)
        x = self.block3(x, edge_feat)
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

        self.input_attention = SpatialAttention()

        # SRCNN Backbone now takes Edge Injection
        # We pass 'edge_feat_dim' as the size of extra channels
        edge_ch = self.edge_feat_dim if use_edge_branch else 0
        self.body = SRCNNBackbone(num_features, edge_channels=edge_ch)

        if scale == 4:
            # Multi-stage upsampling: x2 -> x2
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.SiLU(inplace=True),
                nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.SiLU(inplace=True),
            )
        else:
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
            x = self.input_attention(x)
            
            x = self.body(x, edge_feat)
        else:
            x = self.head(lr)
            x = self.fusion(x)
            x = self.input_attention(x)
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