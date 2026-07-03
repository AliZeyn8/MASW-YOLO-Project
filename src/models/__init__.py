"""
src/models/__init__.py

این فایل ماژول‌های سفارشی MASW-YOLO (MSCA و بعداً AFPN) را در فضای نام
داخلی Ultralytics ثبت می‌کند تا فایل‌های معماری .yaml بتوانند این نام‌ها را بشناسند.
"""

import ultralytics.nn.tasks as tasks
from .modules.msca import MSCA   # import نسبی: از src/models به src/models/modules/msca.py

tasks.MSCA = MSCA

# وقتی afpn.py تکمیل شد:
# from .modules.afpn import CB, BasicBlock, ASFF2, ASFF3
# tasks.CB = CB
# tasks.BasicBlock = BasicBlock
# tasks.ASFF2 = ASFF2
# tasks.ASFF3 = ASFF3

__all__ = ["MSCA"]

print("✅ ماژول‌های سفارشی MASW-YOLO ثبت شدند: MSCA")
