import torch

from wardrobe_seg.review import (
    garment_match_counts,
    garment_threshold_counts,
    select_application_policy,
)


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


def test_application_policy_selects_best_f1_and_its_threshold() -> None:
    reports = [
        {
            "checkpoint": "epoch_04.pt",
            "checkpoint_epoch": 4,
            "selected_confidence_threshold": 0.6,
            "confidence_sweep": {
                "0.6": {"precision": 0.7, "recall": 0.6, "f1": 0.646, "true_positive": 6,
                        "false_positive": 3, "false_negative": 4}
            },
        },
        {
            "checkpoint": "epoch_05.pt",
            "checkpoint_epoch": 5,
            "selected_confidence_threshold": 0.5,
            "confidence_sweep": {
                "0.5": {"precision": 0.68, "recall": 0.64, "f1": 0.659, "true_positive": 7,
                        "false_positive": 3, "false_negative": 4}
            },
        },
    ]
    policy = select_application_policy(reports)
    assert policy["selected_checkpoint_epoch"] == 5
    assert policy["selected_confidence_threshold"] == 0.5


def test_application_policy_rejects_duplicate_epochs() -> None:
    report = {
        "checkpoint": "model.pt",
        "checkpoint_epoch": 4,
        "selected_confidence_threshold": 0.6,
        "confidence_sweep": {"0.6": {"precision": 1.0, "recall": 1.0, "f1": 1.0}},
    }
    try:
        select_application_policy([report, report])
    except ValueError as error:
        assert "Duplicate checkpoint epoch" in str(error)
    else:
        raise AssertionError("Expected duplicate epochs to be rejected")
