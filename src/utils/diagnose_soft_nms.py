"""
diagnose_soft_nms.py

Quick, training-free sanity check for the Soft-NMS integration. Run this in
Colab right after `enable_soft_nms(...)` and BEFORE kicking off a full
training run, to catch the "duplicate boxes keep their original score"
bug (or any regression of it) in seconds instead of 90 epochs.

Usage (Colab):
    !python diagnose_soft_nms.py
"""

import torch

from src.utils.nms import non_max_suppression_soft


def make_prediction(boxes_xywh, obj_scores, num_classes=10, target_class=0):
    """Build a fake YOLO-style raw prediction tensor: (1, 4+nc, N)."""
    n = len(boxes_xywh)
    cls_scores = torch.zeros(n, num_classes)
    cls_scores[:, target_class] = torch.tensor(obj_scores)
    boxes = torch.tensor(boxes_xywh, dtype=torch.float32)
    pred = torch.cat([boxes, cls_scores], dim=1)  # (N, 4+nc)
    return pred.T.unsqueeze(0)  # (1, 4+nc, N)


def main():
    print("=" * 70)
    print("TEST 1: duplicate-box score propagation (gaussian Soft-NMS)")
    print("=" * 70)
    # Two heavily-overlapping boxes (same object, occlusion-style duplicate)
    # + one independent far-away box.
    boxes = [
        [50, 50, 20, 20],   # xywh: center(50,50), 20x20  -> conf 0.90 (the "true" detection)
        [52, 52, 20, 20],   # near-duplicate of box 0     -> conf 0.85 (should be KEPT but DEMOTED)
        [300, 300, 15, 15], # unrelated, far away         -> conf 0.60 (untouched, no overlap)
    ]
    scores = [0.90, 0.85, 0.60]
    pred = make_prediction(boxes, scores)

    # NOTE: we call non_max_suppression_soft directly with soft_nms_* kwargs
    # here (rather than going through enable_soft_nms(), which patches
    # ultralytics.utils.nms.non_max_suppression -- a different call target
    # that only matters when Ultralytics' own train/val loop invokes NMS).
    out = non_max_suppression_soft(
        pred, conf_thres=0.001, iou_thres=0.5, nc=10,
        soft_nms_method="gaussian", soft_nms_sigma=0.5, soft_nms_score_thres=0.001, soft_nms_max_boxes=300,
    )[0]

    print(f"Detections returned: {out.shape[0]} (expect 3 -- Soft-NMS should KEEP the duplicate)")
    for row in out:
        x1, y1, x2, y2, conf, cls = row[:6].tolist()
        print(f"  conf={conf:.4f}  cls={int(cls)}  box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

    n_kept = out.shape[0]
    ok = True
    if n_kept != 3:
        print(f"\n❌ FAIL: expected 3 detections kept, got {n_kept}. "
              f"Soft-NMS should retain the overlapping duplicate, not delete it.")
        ok = False
    else:
        # Identify the duplicate by its box position (near x=50-62), not by
        # sort order, since decay amount can vary. It started at 0.85 and must
        # come back noticeably lower -- if it's still ~0.85, the decayed score
        # isn't being propagated and the precision-collapse bug is back.
        dup_row = min(out.tolist(), key=lambda r: abs(r[0] - 42))  # closest to box1's x1=42
        dup_conf = dup_row[4]
        if dup_conf > 0.7:
            print(f"\n❌ FAIL: duplicate box confidence is {dup_conf:.4f}, "
                  f"barely decayed from its original 0.85. This is the bug that "
                  f"tanks precision/mAP -- the decayed Soft-NMS score is not "
                  f"being written back to the output.")
            ok = False
        else:
            print(f"\n✅ PASS: duplicate box was kept but correctly demoted "
                  f"to conf={dup_conf:.4f} (well below its original 0.85).")

    print()
    print("=" * 70)
    print("TEST 2: hard-NMS baseline sanity (score untouched, duplicate dropped)")
    print("=" * 70)
    out_hard = non_max_suppression_soft(
        pred, conf_thres=0.001, iou_thres=0.5, nc=10,
        soft_nms_method="hard", soft_nms_score_thres=0.001, soft_nms_max_boxes=300,
    )[0]
    print(f"Detections returned: {out_hard.shape[0]} (expect 2 -- hard-NMS deletes the duplicate)")
    for row in out_hard:
        conf = row[4].item()
        print(f"  conf={conf:.4f}")
    if out_hard.shape[0] == 2 and abs(sorted(out_hard[:, 4].tolist())[-1] - 0.90) < 1e-4:
        print("\n✅ PASS: hard-NMS behaves as expected (unaffected by Soft-NMS score-propagation logic).")
    else:
        print("\n❌ FAIL: unexpected hard-NMS behavior -- something else may be broken.")
        ok = False

    print()
    print("=" * 70)
    if ok:
        print("ALL CHECKS PASSED. Safe to launch a full training run.")
    else:
        print("CHECKS FAILED. Do not launch a full training run yet -- "
              "re-copy nms.py from the latest fix and re-run this script.")
    print("=" * 70)


if __name__ == "__main__":
    main()
