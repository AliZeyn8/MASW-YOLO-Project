# """
# src/train.py

# اسکریپت یکپارچه آموزش برای همه آزمایش‌های پروژه MASW-YOLO.

# نحوه اجرا (حتماً از ریشه پروژه، نه از داخل src/):
#     python -m src.train --config configs/exp1_msca.yaml

# قابلیت Resume (برای محیط‌های محدود مثل Colab):
#     اگر یک بار اجرا کرده باشی و session قطع شده باشد، دوباره دقیقاً همین
#     دستور را اجرا کن (همان --config). اسکریپت خودش چک‌پوینت قبلی
#     (weights/last.pt) را در پوشهٔ همان آزمایش پیدا می‌کند و آموزش را از
#     همان‌جا (همان epoch، همان optimizer state) ادامه می‌دهد — نیازی به
#     آرگومان یا تغییر دیگری نیست.
# """

# import argparse
# import os

# import yaml
# from ultralytics import YOLO

# # import نسبی: این خط باعث اجرای src/models/__init__.py می‌شود
# # که ماژول‌های سفارشی (MSCA، AFPN/ASFF2/ASFF3) را در Ultralytics ثبت می‌کند
# from . import models  # noqa: F401

# from .utils.nms import enable_soft_nms

# from .utils.wiou import enable_wiou

# def load_config(exp_config_path):
#     with open('configs/base_config.yaml', 'r') as f:
#         cfg = yaml.safe_load(f)
#     with open(exp_config_path, 'r') as f:
#         exp_cfg = yaml.safe_load(f)
#     cfg.update(exp_cfg)
#     return cfg


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, required=True,
#                          help='مسیر فایل کانفیگ آزمایش، مثال: configs/exp1_msca.yaml')
#     args = parser.parse_args()

#     cfg = load_config(args.config)

#     # ---- چک ایمنی: مطمئن شو مسیر خروجی روی گوگل درایو mount شده ----
#     # (اگر project روی /content/drive/... باشد ولی درایو mount نشده باشد،
#     # این پوشه اصلاً وجود نخواهد داشت و بی‌سروصدا رو دیسک محلی موقت
#     # کولب نوشته می‌شود که با هر قطعی session از بین می‌رود)
#     project_path = cfg['project']
#     if project_path.startswith('/content/drive') and not os.path.isdir('/content/drive/MyDrive'):
#         raise RuntimeError(
#             "❌ مسیر خروجی روی گوگل درایو تنظیم شده ولی درایو mount نشده است.\n"
#             "   قبل از اجرای این اسکریپت، در یک سلول کولب اجرا کن:\n"
#             "       from google.colab import drive\n"
#             "       drive.mount('/content/drive')"
#         )
#     os.makedirs(project_path, exist_ok=True)

#     # ---- اعمال پچ‌های اختیاری؛ باید قبل از ساخت trainer انجام شود ----
#     if cfg.get('soft_nms', False):
#         enable_soft_nms(
#             method=cfg.get('soft_nms_method', 'gaussian'),
#             sigma=cfg.get('soft_nms_sigma', 0.5),
#             score_thres=cfg.get('soft_nms_score_thres', 0.001),
#         )

#     if cfg.get('wiou', False):
#         enable_wiou(
#             alpha=cfg.get('wiou_alpha', 1.9),
#             delta=cfg.get('wiou_delta', 3.0),
#             momentum=cfg.get('wiou_momentum', 0.9999),
#         )
#     # ---- تشخیص چک‌پوینت قبلی برای همین آزمایش (برای Resume) ----
#     checkpoint_path = os.path.join(cfg['project'], cfg['name'], 'weights', 'last.pt')
#     resume = os.path.exists(checkpoint_path)

#     if resume:
#         print(f"🔄 چک‌پوینت قبلی پیدا شد: {checkpoint_path}")
#         print("   آموزش از همان epoch/optimizer state قبلی ادامه می‌یابد ...")
#         model = YOLO(checkpoint_path)
#     elif 'model_yaml' in cfg:
#         model = YOLO(cfg['model_yaml'])
#         model.load(cfg['weights'])
#         print(f"مدل با معماری سفارشی ساخته شد: {cfg['model_yaml']}")
#     else:
#         model = YOLO(cfg['weights'])
#         print(f"مدل مستقیم از وزن‌های آماده بارگذاری شد: {cfg['weights']}")

#     if resume:
#         # طبق مستندات Ultralytics: هنگام resume=True نباید آرگومان‌های
#         # دیگر (epochs، data، ...) را دوباره پاس داد — همه از
#         # args.yaml ذخیره‌شده در پوشهٔ همان آزمایش به‌طور خودکار بازیابی
#         # می‌شوند. پاس‌دادن آرگومان اضافه می‌تواند باعث تناقض/خطا شود.
#         results = model.train(resume=True)
#     else:
#         results = model.train(
#             data=cfg['data'],
#             epochs=cfg['epochs'],
#             batch=cfg['batch'],
#             imgsz=cfg['imgsz'],
#             optimizer=cfg.get('optimizer', 'SGD'),
#             lr0=cfg['lr0'],
#             lrf=cfg['lrf'],
#             momentum=cfg['momentum'],
#             weight_decay=cfg['weight_decay'],
#             device=cfg['device'],
#             workers=cfg['workers'],
#             project=cfg['project'],
#             name=cfg['name'],
#             exist_ok=cfg['exist_ok'],
#             verbose=cfg['verbose'],
#         )

#     print(f"✅ آموزش '{cfg['name']}' با موفقیت به پایان رسید!")
#     print(f"وزن‌های نهایی: {results.save_dir}/weights/best.pt")


# if __name__ == "__main__":
#     main()






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
