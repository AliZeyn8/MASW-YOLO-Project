"""
Custom trainers and validators for MASW-YOLO.

Provides:
    - MASWValidator: Applies Soft-NMS during validation instead of standard NMS.
    - MASWTrainer:   Uses WiseIoULoss for bounding box regression.

Both classes are compatible with Ultralytics v8.4.x.
"""

from copy import copy

import torch
import torch.nn as nn

from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.loss import v8DetectionLoss

from src.utils.loss import WiseIoULoss
from src.utils.nms import soft_nms


# ---------------------------------------------------------------------------
# MASW Validator  (Soft-NMS post-processing)
# ---------------------------------------------------------------------------

class MASWValidator(DetectionValidator):
    """
    MASW-YOLO Validator.

    Extends ``DetectionValidator`` by replacing the standard
    ``non_max_suppression`` call in ``postprocess`` with the custom
    ``soft_nms`` implementation that applies Gaussian score-decay.

    The output format is kept identical to the parent class so the rest
    of the evaluation pipeline (mAP computation, etc.) works unchanged.
    """

    def __init__(self, *args, **kwargs):
        """Initialise the MASWValidator."""
        super().__init__(*args, **kwargs)
        # Soft-NMS parameters (can be overridden via self.args)
        self.soft_sigma = getattr(self.args, "soft_sigma", 0.5)
        self.soft_score_thr = getattr(self.args, "soft_score_thr", 0.001)

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """
        Apply Soft-NMS to prediction outputs.

        Decodes the raw predictions, applies Soft-NMS per image,
        and returns results in the same format as the parent validator.

        Args:
            preds (torch.Tensor): Raw predictions from the model.

        Returns:
            (list[dict[str, torch.Tensor]]): Processed predictions after
                Soft-NMS, where each dict contains 'bboxes', 'conf', 'cls',
                and 'extra' tensors.
        """
        # Use the standard NMS utility first to handle multi-class decoding
        # and class-specific thresholding
        nms_outputs = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            nc=0 if self.args.task == "detect" else self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
        )

        results = []
        for out in nms_outputs:
            if out.shape[0] == 0:
                results.append({
                    "bboxes": torch.zeros((0, 4), device=preds.device),
                    "conf":   torch.zeros(0, device=preds.device),
                    "cls":    torch.zeros(0, device=preds.device),
                    "extra":  torch.zeros((0, 0), device=preds.device),
                })
                continue

            boxes = out[:, :4]   # xyxy
            conf  = out[:, 4]    # confidence
            cls   = out[:, 5]    # class index
            extra = out[:, 6:]   # any extra fields

            # Apply Soft-NMS on the already-filtered predictions
            # This re-ranks boxes with Gaussian decay
            kept_boxes, kept_conf = soft_nms(
                boxes, conf,
                sigma=self.soft_sigma,
                score_threshold=self.soft_score_thr,
                iou_threshold=self.args.iou,
            )

            if kept_boxes.shape[0] == 0:
                results.append({
                    "bboxes": torch.zeros((0, 4), device=preds.device),
                    "conf":   torch.zeros(0, device=preds.device),
                    "cls":    torch.zeros(0, device=preds.device),
                    "extra":  torch.zeros((0, 0), device=preds.device),
                })
                continue

            # Map kept boxes back to original indices to retrieve class labels
            # Since soft_nms may keep boxes in a different order, we match
            # by computing IoU between kept boxes and original boxes.
            # A simpler approach: keep the full output and just update scores
            # from soft_nms, then re-filter by score threshold.
            # For robustness, we re-run the matching:
            kept_indices = []
            for kb in kept_boxes:
                ious = ops.box_iou(kb.unsqueeze(0), boxes)[0]
                match = ious.argmax()
                if ious[match] > 0.5:
                    kept_indices.append(match)
                else:
                    kept_indices.append(-1)

            valid = [i for i in kept_indices if i >= 0]
            if not valid:
                results.append({
                    "bboxes": torch.zeros((0, 4), device=preds.device),
                    "conf":   torch.zeros(0, device=preds.device),
                    "cls":    torch.zeros(0, device=preds.device),
                    "extra":  torch.zeros((0, 0), device=preds.device),
                })
                continue

            idx = torch.tensor(valid, device=preds.device)
            # Re-sort by confidence (descending)
            order = kept_conf[valid].argsort(descending=True)
            results.append({
                "bboxes": kept_boxes[valid][order],
                "conf":   kept_conf[valid][order],
                "cls":    cls[idx][order],
                "extra":  extra[idx][order] if extra.shape[1] > 0
                          else torch.zeros((len(valid), 0), device=preds.device),
            })

        return results


# ---------------------------------------------------------------------------
# MASW Trainer  (WiseIoU loss)
# ---------------------------------------------------------------------------

class MASWTrainer(DetectionTrainer):
    """
    MASW-YOLO Trainer.

    Extends ``DetectionTrainer`` by:
        1. Replacing the standard CIoU-based ``BboxLoss`` with
           ``WiseIoULoss`` in the training criterion.
        2. Using ``MASWValidator`` instead of the default validator so
           Soft-NMS is applied during evaluation.

    Usage:
        trainer = MASWTrainer(overrides={"model": "model.yaml", "data": "data.yaml"})
        trainer.train()
    """

    def __init__(
        self,
        cfg="default.yaml",
        overrides: dict | None = None,
        _callbacks: dict | None = None,
    ):
        """Initialise the MASWTrainer."""
        super().__init__(cfg, overrides, _callbacks)

        # Soft-NMS parameters passed to validator
        if overrides is not None:
            self.args.soft_sigma = overrides.get("soft_sigma", 0.5)
            self.args.soft_score_thr = overrides.get("soft_score_thr", 0.001)

    def get_validator(self):
        """
        Return a ``MASWValidator`` (instead of the default ``DetectionValidator``).

        Overrides ``DetectionTrainer.get_validator``.
        """
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return MASWValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def _setup_train(self):
        """
        Set up training and inject ``WiseIoULoss`` into the model's criterion.

        Calls the parent ``_setup_train`` (which creates the model, loads
        the criterion, etc.) and then replaces the standard ``BboxLoss``
        inside ``model.criterion.bbox_loss`` with a ``WiseIoULoss`` instance
        that preserves the same ``reg_max`` and DFL settings.
        """
        super()._setup_train()

        # ---- Inject WiseIoULoss into the criterion ----
        # The criterion (v8DetectionLoss) was created by the model during
        # setup_model.  It contains self.bbox_loss (a BboxLoss instance).
        # We replace it with our WiseIoULoss, copying the reg_max.
        if hasattr(self.model, "criterion") and self.model.criterion is not None:
            criterion = self.model.criterion
            if hasattr(criterion, "bbox_loss"):
                old_bbox_loss = criterion.bbox_loss
                reg_max = old_bbox_loss.dfl_loss.reg_max if old_bbox_loss.dfl_loss else 16

                # Create the WiseIoU replacement
                wise_loss = WiseIoULoss(
                    reg_max=reg_max,
                    version=getattr(self.args, "wiou_version", "v3"),
                    beta=getattr(self.args, "wiou_beta", 1.0),
                    delta=getattr(self.args, "wiou_delta", 0.5),
                ).to(old_bbox_loss.dfl_loss.weight.device if old_bbox_loss.dfl_loss else self.device)

                criterion.bbox_loss = wise_loss

                # Log the replacement
                from ultralytics.utils import LOGGER
                LOGGER.info(
                    f"MASW-YOLO: Replaced BboxLoss with WiseIoULoss "
                    f"(version={wise_loss.version}, beta={wise_loss.beta}, delta={wise_loss.delta})"
                )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Build the model from a YAML config and optionally load pretrained weights.

        This override adds support for the custom ``MASWDetectionModel``
        (which registers MSCA/AFPN layers).  If the YAML references standard
        Ultralytics layers only, the default model is used.

        Args:
            cfg (str | dict): Path to YAML config or parsed dict.
            weights (str | None): Path to pretrained weights.
            verbose (bool): Whether to print layer info.

        Returns:
            (nn.Module): The detection model.
        """
        # Try to use the custom MASWDetectionModel if available
        try:
            from src.models.custom_model import MASWDetectionModel
            model = MASWDetectionModel(cfg=cfg, nc=self.args.nc, verbose=verbose)
            if weights:
                model.load(weights)
            return model
        except Exception:
            # Fall back to the default behaviour
            return super().get_model(cfg, weights, verbose)
