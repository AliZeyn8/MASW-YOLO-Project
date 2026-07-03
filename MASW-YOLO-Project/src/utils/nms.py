"""
Non-Maximum Suppression utilities for MASW-YOLO.

Implements Soft-NMS as a post-processing step to reduce false positives
in dense small-object scenarios. Unlike standard NMS, Soft-NMS decays
the confidence scores of overlapping boxes rather than discarding them
completely, which helps retain true positives for small and occluded objects.

Equation (5) of the paper:
    s_i = s_i * exp( - iou(M, b_i)^2 / sigma )

Reference:
    "Soft-NMS — Improving Object Detection With One Line of Code"
    https://arxiv.org/abs/1704.04503
"""

import torch


def _compute_iou(boxes: torch.Tensor, i: int) -> torch.Tensor:
    """
    Compute IoU between box `i` (index) and all boxes in `boxes`.

    Args:
        boxes (torch.Tensor): (N, 4) in xyxy format.
        i (int): Index of the reference box.

    Returns:
        torch.Tensor: IoU values of shape (N,).
    """
    # Reference box
    x1_i, y1_i, x2_i, y2_i = boxes[i].unbind()
    area_i = (x2_i - x1_i) * (y2_i - y1_i)

    # Intersection
    x1 = torch.max(boxes[:, 0], x1_i)
    y1 = torch.max(boxes[:, 1], y1_i)
    x2 = torch.min(boxes[:, 2], x2_i)
    y2 = torch.min(boxes[:, 3], y2_i)

    w = (x2 - x1).clamp(min=0)
    h = (y2 - y1).clamp(min=0)
    inter = w * h

    # Area of all boxes
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = area + area_i - inter
    return inter / (union + 1e-7)


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
    iou_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Soft-NMS with Gaussian decay.

    Iteratively picks the highest-scoring box, decays scores of all other
    boxes that overlap with it, and repeats until all remaining boxes are
    below the score threshold.

    The decay follows the paper's Gaussian formulation (Equation 5):
        s_i = s_i * exp( - iou(M, b_i)^2 / sigma )

    Args:
        boxes (torch.Tensor): Bounding boxes, shape (N, 4), xyxy format.
        scores (torch.Tensor): Confidence scores, shape (N,).
        sigma (float): Sigma parameter for Gaussian decay. Default: 0.5.
        score_threshold (float): Minimum score to keep a box. Default: 0.001.
        iou_threshold (float): Not used directly in Gaussian soft-NMS (decay
            is continuous), but kept for API compatibility. Default: 0.5.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - kept_boxes: (M, 4) filtered boxes.
            - kept_scores: (M,) updated scores after decay.
    """
    if boxes.numel() == 0:
        return boxes, scores

    N = boxes.shape[0]
    # Work on copies
    boxes = boxes.clone()
    scores = scores.clone()

    # Indices of boxes that are still alive
    indices = torch.arange(N, device=boxes.device)
    # Output collections
    keep = []
    keep_scores = []

    while indices.numel() > 0:
        # Pick the box with the highest score
        max_score_idx = scores[indices].argmax()
        max_idx = indices[max_score_idx]

        # Stop if the best remaining score is below threshold
        if scores[max_idx] < score_threshold:
            break

        keep.append(max_idx)
        keep_scores.append(scores[max_idx])

        # Remove the picked box from indices
        indices = torch.cat([indices[:max_score_idx], indices[max_score_idx + 1:]])

        if indices.numel() == 0:
            break

        # Compute IoU between the picked box and all remaining boxes
        ious = _compute_iou(boxes, max_idx)[indices]

        # Gaussian decay:  s_i = s_i * exp( - iou^2 / sigma )
        decay = torch.exp(-(ious ** 2) / sigma)
        scores[indices] = scores[indices] * decay

    if len(keep) == 0:
        return torch.zeros((0, 4), device=boxes.device), torch.zeros(0, device=boxes.device)

    kept_boxes = boxes[torch.tensor(keep, device=boxes.device)]
    kept_scores = torch.tensor(keep_scores, device=boxes.device)
    return kept_boxes, kept_scores
