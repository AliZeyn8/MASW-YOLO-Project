"""
Asymptotic Feature Pyramid Network (AFPN) Module.

Implements the AFPN neck for multi-scale feature fusion as described in the
MASW-YOLO paper. AFPN progressively fuses features from adjacent layers
using three key building blocks:

    1. CB (Cross-stage Block): A cross-stage feature transformation that
       preserves and merges gradient flow across layers.
    2. BB (Bottom-up Block): Bottom-up feature aggregation that propagates
       semantic information from higher resolutions.
    3. ASFF (Adaptively Spatial Feature Fusion): An adaptive fusion mechanism
       that learns spatial importance weights for each scale.

The progressive fusion strategy fuses adjacent-level features step by step,
avoiding the semantic gaps caused by fusing non-adjacent features directly.

Reference:
    MASW-YOLO: Improved YOLOv8 for UAV Small Object Detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CB(nn.Module):
    """
    Cross-stage Block (CB).

    A building block for cross-stage feature transformation that helps
    preserve gradient flow and reduce redundant gradient information.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        hidden_ratio (float): Ratio for the hidden dimension. Default: 0.5.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_ratio: float = 0.5):
        super().__init__()
        hidden_channels = int(in_channels * hidden_ratio)

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

        # Shortcut connection
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CB.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after cross-stage transformation.
        """
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.conv3(out)
        out = self.bn3(out)
        return self.act(out + identity)


class BB(nn.Module):
    """
    Bottom-up Block (BB).

    Aggregates features from bottom-up pathway, fusing higher-resolution
    (shallower) features with lower-resolution (deeper) ones.

    Args:
        channels (int): Number of channels for both inputs (already projected).
    """

    def __init__(self, channels: int):
        super().__init__()

        # Fusion: concatenate -> 1x1 reduce -> 3x3 refine
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, low_feat: torch.Tensor, high_feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of BB.

        Args:
            low_feat (torch.Tensor): Higher-resolution (finer) feature map.
            high_feat (torch.Tensor): Lower-resolution (coarser) feature map.

        Returns:
            torch.Tensor: Fused feature map at the higher resolution.
        """
        # Upsample coarser features to finer resolution
        if high_feat.shape[2:] != low_feat.shape[2:]:
            high_feat = F.interpolate(
                high_feat, size=low_feat.shape[2:],
                mode="bilinear", align_corners=False,
            )

        out = torch.cat([low_feat, high_feat], dim=1)
        out = self.fusion_conv(out)
        return out


class ASFF(nn.Module):
    """
    Adaptively Spatial Feature Fusion (ASFF).

    Learns adaptive spatial weights for fusing features from different scales.
    Each spatial location gets a weight for each scale, allowing the network
    to focus on the most informative scale for each region.

    Args:
        num_scales (int): Number of input scales to fuse.
        channels (int): Number of channels for each scale (all must match).
    """

    def __init__(self, num_scales: int = 3, channels: int = 256):
        super().__init__()
        self.num_scales = num_scales

        # Learnable weight parameters (normalised by softmax across scales)
        self.scale_weights = nn.Parameter(
            torch.ones(num_scales, 1, 1, 1) / num_scales
        )

        # Post-fusion refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ASFF.

        Args:
            *features: Variable-length list of feature maps from different scales.
                      All must have the same channels and spatial size.

        Returns:
            torch.Tensor: Adaptively fused feature map.
        """
        # Softmax normalisation over scales
        weight_norm = F.softmax(self.scale_weights, dim=0)

        # Weighted sum
        fused = sum(w * feat for w, feat in zip(weight_norm, features))

        return self.refine(fused)


class AFPN(nn.Module):
    """
    Asymptotic Feature Pyramid Network (AFPN).

    Accepts multi-scale features from the backbone (e.g., [C2, C3, C4, C5])
    and performs progressive adjacent-level fusion:

        f1 = fuse(C2, C3)
        f2 = fuse(f1, C4)
        f3 = fuse(f2, C5)

    At each fusion step a Bottom-up Block (BB) is used, and all outputs are
    finally aggregated with an ASFF module.

    Args:
        in_channels_list (list): Input channels for each scale, ordered from
            highest resolution to lowest, e.g. [64, 128, 256, 512] for
            [C2, C3, C4, C5].
        out_channels (int): Common channel dimension for all fused outputs.
    """

    def __init__(self, in_channels_list: list, out_channels: int = 256):
        super().__init__()

        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.num_scales = len(in_channels_list)

        # 1x1 projections to common channel dimension
        self.projections = nn.ModuleList()
        for in_ch in in_channels_list:
            self.projections.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                )
            )

        # BB blocks for progressive fusion (one per adjacent pair)
        self.bb_blocks = nn.ModuleList(
            [BB(out_channels) for _ in range(self.num_scales - 1)]
        )

        # ASFF at the top for adaptive multi-scale fusion
        self.asff = ASFF(num_scales=self.num_scales, channels=out_channels)

    def forward(self, features: list) -> torch.Tensor:
        """
        Forward pass of AFPN.

        Accepts a list of feature maps from the backbone (highest resolution
        first) and progressively fuses them from adjacent levels:

            f1 = BB(proj[C2], proj[C3])   -- fuse C2+C3
            f2 = BB(f1,        proj[C4])  -- fuse result with C4
            f3 = BB(f2,        proj[C5])  -- fuse result with C5

        All intermediate results are resized to the finest resolution and
        aggregated via ASFF.

        Args:
            features (list): Multi-scale feature maps, e.g. [C2, C3, C4, C5]
                ordered from highest resolution to lowest.

        Returns:
            torch.Tensor: Fused output after progressive fusion.
        """
        assert len(features) == self.num_scales, (
            f"Expected {self.num_scales} feature maps, got {len(features)}"
        )

        # Project all scales to a common channel dimension
        proj = [proj_fn(f) for proj_fn, f in zip(self.projections, features)]

        # Progressive fusion: adjacent levels step-by-step
        # f1 = fuse(C2, C3), then f2 = fuse(f1, C4), then f3 = fuse(f2, C5)
        fused = proj[0]
        for i in range(self.num_scales - 1):
            fused = self.bb_blocks[i](proj[i + 1], fused)

        # Collect all intermediate outputs for ASFF (resize to finest)
        resized = []
        target_size = proj[0].shape[2:]  # finest resolution
        for f in proj:
            if f.shape[2:] != target_size:
                f = F.interpolate(
                    f, size=target_size,
                    mode="bilinear", align_corners=False,
                )
            resized.append(f)

        # Resize fused output as well (it may already match)
        if fused.shape[2:] != target_size:
            fused = F.interpolate(
                fused, size=target_size,
                mode="bilinear", align_corners=False,
            )

        # Combine all scales via ASFF
        output = self.asff(*resized)

        return output
