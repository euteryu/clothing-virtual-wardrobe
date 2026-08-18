from wardrobe_seg.data import select_class_aware_ids, split_image_ids


def test_split_is_disjoint_complete_and_deterministic() -> None:
    ids = [f"image-{index}" for index in range(20)]
    train = split_image_ids(ids, "train", val_fraction=0.2, seed=7)
    val = split_image_ids(ids, "val", val_fraction=0.2, seed=7)
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(ids)
    assert val == split_image_ids(list(reversed(ids)), "val", val_fraction=0.2, seed=7)


def test_class_aware_subset_covers_rare_class_and_is_deterministic() -> None:
    rows = {
        "common-1": [{"ClassId": "0"}],
        "common-2": [{"ClassId": "0"}],
        "common-3": [{"ClassId": "0"}],
        "rare": [{"ClassId": "1"}],
    }
    selected = select_class_aware_ids(list(rows), rows, max_images=2, seed=11)
    assert "rare" in selected
    assert selected == select_class_aware_ids(list(reversed(rows)), rows, max_images=2, seed=11)
