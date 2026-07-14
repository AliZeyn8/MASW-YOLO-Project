"""
src/utils/wiou.py

Faithful implementation of Wise-IoU v3 (Tong et al., 2023) exactly as
described in the MASW-YOLO paper, Eqs. (6)-(12):

    L_CIoU = L_IoU + (x-xgt)^2+(y-ygt)^2 / (Wg^2+Hg^2) + alpha*v      (6)
    alpha  = v / (L_IoU + v)                                          (7)
    v      = (4/pi^2) * (arctan(w/h) - arctan(wgt/hgt))^2             (8)
    L_IoU  = 1 - IoU                                                  (9)   <- area form Wi*Hi/(...)  is just IoU
    L_WIoU = r * R_WIoU * L_IoU,  R_WIoU in [1,e), L_IoU in [0,1]     (10)
    R_WIoU = exp( ((x-xgt)^2+(y-ygt)^2) / (Wg^2+Hg^2) )               (11)
    r      = beta / (delta * alpha_hp^(beta-delta)),  beta = L_IoU / L_IoU_bar   (12)

This file is 100% self-contained. It does NOT import or touch
src/models/modules/afpn.py, msca.py, or src/utils/nms.py, and it does
not modify the network graph in any way -- it only replaces the
box-regression loss term. Consequently it cannot change Params/M or
FLOPs/G (see Table 2: the WIoU-only row is 8.1 G / 3.01 M, identical
to the untouched YOLOv8n baseline).

Usage (drop-in, same pattern as your existing exp5_wiou.yaml):

    from src.utils.wiou import enable_wiou
    enable_wiou(alpha=1.9, delta=3.0, momentum=0.9999)
    # must be called BEFORE model.train()/model.val() build the
    # criterion (v8DetectionLoss), since BboxLoss is resolved by name
    # at construction time.

    ... model.train(...) ...

    from src.utils.wiou import disable_wiou
    disable_wiou()   # restores stock CIoU BboxLoss if needed
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.tal import bbox2dist


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _center_dist_sq(boxes1, boxes2):
    """(x - xgt)^2 + (y - ygt)^2, boxes in xyxy."""
    c1x = (boxes1[..., 0] + boxes1[..., 2]) / 2
    c1y = (boxes1[..., 1] + boxes1[..., 3]) / 2
    c2x = (boxes2[..., 0] + boxes2[..., 2]) / 2
    c2y = (boxes2[..., 1] + boxes2[..., 3]) / 2
    return (c1x - c2x) ** 2 + (c1y - c2y) ** 2


def _iou(boxes1, boxes2, eps=1e-7):
    """Standard IoU (Eq. 9: L_IoU = 1 - IoU)."""
    inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - inter
    return inter / (union + eps)


def _enclose_wh_sq(boxes1, boxes2, eps=1e-7):
    """Wg^2 + Hg^2 of the smallest enclosing box (used by both Eq. 6 and Eq. 11)."""
    ex1 = torch.min(boxes1[..., 0], boxes2[..., 0])
    ey1 = torch.min(boxes1[..., 1], boxes2[..., 1])
    ex2 = torch.max(boxes1[..., 2], boxes2[..., 2])
    ey2 = torch.max(boxes1[..., 3], boxes2[..., 3])
    wg = (ex2 - ex1).clamp(min=eps)
    hg = (ey2 - ey1).clamp(min=eps)
    return wg ** 2 + hg ** 2


# --------------------------------------------------------------------------- #
# WIoU v3 core (Eqs. 10-12)
# --------------------------------------------------------------------------- #
class WiseIoUv3:
    """
    Stateful WIoU-v3. Holds the running (EMA) average L_IoU_bar across
    the whole training run, as required by Eq. (12) -- this is the
    "dynamic sliding average" mentioned in the paper. It is NOT a
    per-batch statistic, so it must persist across forward calls
    (hence a small stateful object rather than a pure function).
    """

    def __init__(self, alpha: float = 1.9, delta: float = 3.0, momentum: float = 0.9999):
        # alpha, delta: fixed hyper-parameters from Eq. (12).
        # The paper does not pin exact values; alpha=1.9, delta=3.0 are
        # the values recommended by the original Wise-IoU v3 paper
        # (Tong et al. 2023), which this paper's Eq. (12) reproduces.
        self.alpha = alpha
        self.delta = delta
        self.momentum = momentum
        self._iou_bar = None  # scalar running mean of L_IoU, lazily initialised

    def __call__(self, pred_boxes, target_boxes, eps: float = 1e-7):
        """
        pred_boxes, target_boxes: (..., 4) xyxy tensors.
        Returns per-anchor WIoU loss, same leading shape as input minus
        last dim (i.e. one scalar loss per box pair).
        """
        iou = _iou(pred_boxes, target_boxes, eps)
        l_iou = 1.0 - iou  # Eq. (9)

        # --- update / read the dynamic sliding average L_IoU_bar ---
        with torch.no_grad():
            batch_mean = l_iou.mean()
            if self._iou_bar is None:
                self._iou_bar = batch_mean.clone()
            else:
                self._iou_bar.mul_(self.momentum).add_(batch_mean, alpha=1 - self.momentum)
            iou_bar = self._iou_bar.clamp(min=eps)

            # beta = L_IoU / L_IoU_bar  (Eq. 12), detached: r must not
            # receive gradient, only re-weight L_IoU's own gradient.
            beta = (l_iou.detach() / iou_bar).clamp(min=0.0)
            r = beta / (self.delta * torch.pow(self.alpha, beta - self.delta) + eps)

        # R_WIoU (Eq. 11): distance-focusing term, amplifies L_IoU of
        # ordinary/modest anchors. Kept differentiable (contributes
        # gradient, unlike r).
        r_wiou = torch.exp(_center_dist_sq(pred_boxes, target_boxes) / _enclose_wh_sq(pred_boxes, target_boxes, eps))

        loss = r * r_wiou * l_iou  # Eq. (10)
        return loss

    def state_dict(self):
        return {"iou_bar": None if self._iou_bar is None else self._iou_bar.item()}

    def load_state_dict(self, state, device="cpu"):
        v = state.get("iou_bar")
        self._iou_bar = None if v is None else torch.tensor(v, device=device)


# --------------------------------------------------------------------------- #
# Drop-in replacement for ultralytics.utils.loss.BboxLoss
# --------------------------------------------------------------------------- #
class WiseIoULoss(BboxLoss):
    def __init__(self, reg_max=16, alpha: float = 1.9, delta: float = 3.0, momentum: float = 0.9999):
        super().__init__(reg_max)
        self.wiou = WiseIoUv3(alpha=alpha, delta=delta, momentum=momentum)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
                target_scores_sum, fg_mask, imgsz, stride):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        loss = self.wiou(pred_bboxes[fg_mask], target_bboxes[fg_mask]).unsqueeze(-1)
        loss_iou = (loss * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight)
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


# --------------------------------------------------------------------------- #
# Global enable/disable, mirroring src/utils/nms.py's enable_soft_nms pattern
# --------------------------------------------------------------------------- #
import ultralytics.utils.loss as _ultra_loss
from ultralytics.utils import LOGGER


def enable_wiou(alpha: float = 1.9, delta: float = 3.0, momentum: float = 0.9999):
    """
    Monkey-patches ultralytics.utils.loss.BboxLoss -> WiseIoULoss.
    Must run before model.train()/model.val() build v8DetectionLoss.
    Independent of AFPN/MSCA/Soft-NMS: only replaces the box-loss term,
    never touches nn.Module graph, so Params/M and FLOPs/G are unaffected.
    """
    def _factory(reg_max: int = 16, *args, **kwargs):
        return WiseIoULoss(reg_max, alpha=alpha, delta=delta, momentum=momentum)

    _ultra_loss.BboxLoss = _factory
    LOGGER.info(f"✅ WIoU-v3 enabled (alpha={alpha}, delta={delta}, momentum={momentum})")


def disable_wiou():
    import importlib
    importlib.reload(_ultra_loss)
    LOGGER.info("↩️ Standard Ultralytics CIoU BboxLoss restored.")
