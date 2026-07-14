import torch
import torch.nn as nn
import torch.nn.functional as F


def _resize_to(x: torch.Tensor, size) -> torch.Tensor:
    if x.shape[-2:] == tuple(size):
        return x
    if x.shape[-2] < size[0]:
        return F.interpolate(x, size=size, mode="nearest")
    return F.adaptive_avg_pool2d(x, size)


class ASFF2(nn.Module):
    def __init__(self, c1: int, c2: int, c_out: int = None, level: int = 0):
        super().__init__()
        c_out = c_out or c1
        self.level = level

        self.proj0 = nn.Conv2d(c1, c_out, 1) if c1 != c_out else nn.Identity()
        self.proj1 = nn.Conv2d(c2, c_out, 1) if c2 != c_out else nn.Identity()

        self.weight0 = nn.Conv2d(c_out, 1, 1)
        self.weight1 = nn.Conv2d(c_out, 1, 1)

        self.conv_out = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x0, x1 = x
        x0 = self.proj0(x0)
        x1 = self.proj1(x1)

        target = x0.shape[-2:] if self.level == 0 else x1.shape[-2:]
        x0 = _resize_to(x0, target)
        x1 = _resize_to(x1, target)

        w = torch.cat([self.weight0(x0), self.weight1(x1)], dim=1)
        w = torch.softmax(w, dim=1)

        fused = x0 * w[:, 0:1] + x1 * w[:, 1:2]
        return self.act(self.bn(self.conv_out(fused)))


class ASFF3(nn.Module):
    def __init__(self, c1: int, c2: int, c3: int, c_out: int = None, level: int = 0):
        super().__init__()
        c_out = c_out or c1
        self.level = level

        self.proj0 = nn.Conv2d(c1, c_out, 1) if c1 != c_out else nn.Identity()
        self.proj1 = nn.Conv2d(c2, c_out, 1) if c2 != c_out else nn.Identity()
        self.proj2 = nn.Conv2d(c3, c_out, 1) if c3 != c_out else nn.Identity()

        self.weight0 = nn.Conv2d(c_out, 1, 1)
        self.weight1 = nn.Conv2d(c_out, 1, 1)
        self.weight2 = nn.Conv2d(c_out, 1, 1)

        self.conv_out = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x0, x1, x2 = x
        x0 = self.proj0(x0)
        x1 = self.proj1(x1)
        x2 = self.proj2(x2)

        targets = [x0.shape[-2:], x1.shape[-2:], x2.shape[-2:]]
        target = targets[self.level]
        x0 = _resize_to(x0, target)
        x1 = _resize_to(x1, target)
        x2 = _resize_to(x2, target)

        w = torch.cat([self.weight0(x0), self.weight1(x1), self.weight2(x2)], dim=1)
        w = torch.softmax(w, dim=1)

        fused = x0 * w[:, 0:1] + x1 * w[:, 1:2] + x2 * w[:, 2:3]
        return self.act(self.bn(self.conv_out(fused)))


class BasicBlock(nn.Module):
    def __init__(self, c1: int, c2: int = None, shortcut: bool = True):
        super().__init__()
        c2 = c2 or c1
        self.cv1 = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.cv2 = nn.Conv2d(c2, c2, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.act(self.bn1(self.cv1(x)))
        y = self.bn2(self.cv2(y))
        return self.act(x + y) if self.add else self.act(y)
