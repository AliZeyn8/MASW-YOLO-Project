# """
# Multi-Scale Convolution Attention (MSCA) Module.

# Implements the MSCA block as described in the MASW-YOLO paper.
# The module consists of three main steps:
#     1. Depthwise Convolution (5x5) for spatial feature extraction.
#     2. Multi-branch Strip Convolution to capture multi-scale contextual
#        information using parallel strip kernel pairs
#        (1x7/7x1, 1x11/11x1, 1x21/21x1), all summed together.
#     3. Point-wise Convolution (1x1 Conv) to produce the attention map.

#     Output: Att * F (element-wise multiplication with original input).

# Reference:
#     MASW-YOLO: Improved YOLOv8 for UAV Small Object Detection.
# """

# import torch
# import torch.nn as nn


# class MSCA(nn.Module):
#     """
#     Multi-Scale Convolution Attention module.

#     Given an input feature map F:
#         Step 1: 5x5 Depthwise Conv → F'
#         Step 2: Parallel strip conv pairs on F', summed → F''
#         Step 3: 1x1 Conv → attention map Att
#         Output: Att ⊙ F

#     Args:
#         channels (int): Number of input/output channels.
#     """

#     def __init__(self, channels: int):
#         super().__init__()

#         # ---------------------------------------------------------------
#         # Step 1: 5x5 Depthwise Convolution
#         # ---------------------------------------------------------------
#         self.deep_conv = nn.Conv2d(
#             in_channels=channels,
#             out_channels=channels,
#             kernel_size=5,
#             padding=2,
#             groups=channels,  # depthwise: one filter per channel
#             bias=False,
#         )
#         self.deep_bn = nn.BatchNorm2d(channels)

#         # ---------------------------------------------------------------
#         # Step 2: Multi-branch Strip Convolution
#         # Three parallel branches, each a pair of asymmetric (strip) convs:
#         #   (1×7, 7×1), (1×11, 11×1), (1×21, 21×1)
#         # All branch outputs are summed element-wise.
#         # ---------------------------------------------------------------
#         strip_kernels = [7, 11, 21]
#         self.strip_branches = nn.ModuleList()
#         for k in strip_kernels:
#             branch = nn.Sequential(
#                 nn.Conv2d(
#                     channels, channels,
#                     kernel_size=(1, k), padding=(0, k // 2),
#                     groups=channels, bias=False,
#                 ),
#                 nn.Conv2d(
#                     channels, channels,
#                     kernel_size=(k, 1), padding=(k // 2, 0),
#                     groups=channels, bias=False,
#                 ),
#             )
#             self.strip_branches.append(branch)

#         # ---------------------------------------------------------------
#         # Step 3: Point-wise Convolution (1x1) → attention map
#         # ---------------------------------------------------------------
#         self.point_conv = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.SiLU(inplace=True),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of MSCA.

#         Args:
#             x (torch.Tensor): Input tensor of shape (B, C, H, W).

#         Returns:
#             torch.Tensor: Output tensor of shape (B, C, H, W).
#         """
#         # Step 1: 5x5 Depthwise Convolution
#         out = self.deep_conv(x)
#         out = self.deep_bn(out)

#         # Step 2: Parallel strip convolutions summed
#         strip_out = 0
#         for branch in self.strip_branches:
#             strip_out = strip_out + branch(out)

#         # Step 3: 1x1 Convolution → attention map
#         att = self.point_conv(strip_out)

#         # Element-wise multiplication with the original input
#         return x * att


"""
models/modules/msca.py

پیاده‌سازی Multi-Scale Convolutional Attention (MSCA)
منطبق بر معادلات (1) و (2) و شکل ۲ مقاله MASW-YOLO

این ماژول درست بعد از لایه SPPF در انتهای backbone قرار می‌گیرد (طبق شکل ۱ مقاله)
و ورودی/خروجی آن شکل یکسانی دارند (فقط ویژگی‌ها را بازوزن‌دهی می‌کند).
"""

import torch
import torch.nn as nn


class MSCA(nn.Module):
    def __init__(self, c1):
        """
        c1: تعداد کانال‌های ورودی/خروجی
            (وقتی در فایل yaml معماری استفاده می‌شود، Ultralytics این عدد را
            خودکار از خروجی لایه قبلی پر می‌کند)
        """
        super().__init__()
        dim = c1

        # مرحله ۱ (شکل ۲: جعبه سبز بالا "Multi-scale Feature")
        # کانولوشن عمقی ۵×۵ روی هر کانال جداگانه (groups=dim یعنی depthwise conv)
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        # مرحله ۲ (شکل ۲: سه شاخه بنفش موازی)
        # هر شاخه یک کانولوشن n×n را با دو کانولوشن نواری 1×n و n×1 شبیه‌سازی می‌کند
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7),  padding=(0, 3),  groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1),  padding=(3, 0),  groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5),  groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0),  groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        # مرحله ۳ (شکل ۲: جعبه سبز پایین "Channel Mixing")
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()                # نگه‌داشتن ورودی خام برای ضرب نهایی
        attn = self.conv0(x)         # خروجی مرحله ۱

        attn_0 = self.conv0_2(self.conv0_1(attn))   # شاخه با عرض ۷
        attn_1 = self.conv1_2(self.conv1_1(attn))   # شاخه با عرض ۱۱
        attn_2 = self.conv2_2(self.conv2_1(attn))   # شاخه با عرض ۲۱

        attn = attn + attn_0 + attn_1 + attn_2       # جمع سه شاخه با خروجی مرحله ۱ (⊕ در شکل ۲)
        attn = self.conv3(attn)                      # تولید نقشه توجه نهایی (Att)

        return attn * u                               # Out = Att ⊗ F  (معادله ۲)