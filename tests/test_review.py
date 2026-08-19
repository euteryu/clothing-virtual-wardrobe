import torch

from wardrobe_seg.review import garment_match_counts, garment_threshold_counts


def test_garment_matching_respects_class_score_and_iou() -> None:
    masks = torch.zeros((3, 4, 4), dtype=torch.uint8)
    masks[0, :2, :2] = 1
    masks[1, :2, :2] = 1
    masks[2, 2:, 2:] = 1
    target_masks = torch.zeros((2, 4, 4), dtype=torch.uint8)
    target_masks[0, :2, :2] = 1
    target_masks[1, 2:, 2:] = 1
    counts = garment_match_counts(
        pred_labels=torch.tensor([1, 2, 1]),
        pred_scores=torch.tensor([0.9, 0.8, 0.4]),
        pred_masks=masks,
        target_labels=torch.tensor([1, 1]),
        target_masks=target_masks,
        threshold=0.5,
    )
    assert counts == (1, 1, 1)
    sweep = garment_threshold_counts(
        pred_labels=torch.tensor([1, 2, 1]),
        pred_scores=torch.tensor([0.9, 0.8, 0.4]),
        pred_masks=masks,
        target_labels=torch.tensor([1, 1]),
        target_masks=target_masks,
        thresholds=(0.5, 0.85),
    )
    assert sweep[0.5] == (1, 1, 1)
    assert sweep[0.85] == (1, 0, 1)
