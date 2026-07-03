# MASW-YOLO Model Modules
# Contains attention mechanisms, feature pyramid networks, and other building blocks.

# from .msca import MSCA
# from .afpn import AFPN

# __all__ = ["MSCA", "AFPN"]

"""
models/__init__.py

این فایل، ماژول‌های سفارشی MASW-YOLO (MSCA و بعداً AFPN) را در فضای نام
داخلی Ultralytics ثبت می‌کند، تا فایل‌های معماری .yaml بتوانند این نام‌ها را بشناسند.

نحوه استفاده: در ابتدای train.py فقط بنویسید:
    import models   # همین کافی است، هیچ فراخوانی دیگری لازم نیست
"""

import ultralytics.nn.tasks as tasks
from .modules.msca import MSCA

# ثبت MSCA در فضای نام Ultralytics
tasks.MSCA = MSCA

# --- وقتی AFPN را در مرحله بعد ساختیم، این بخش را از کامنت خارج می‌کنیم ---
# from .modules.afpn import CB, BasicBlock, ASFF2, ASFF3
# tasks.CB = CB
# tasks.BasicBlock = BasicBlock
# tasks.ASFF2 = ASFF2
# tasks.ASFF3 = ASFF3

__all__ = ["MSCA"]

print("✅ ماژول‌های سفارشی MASW-YOLO ثبت شدند: MSCA")