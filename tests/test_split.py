from wardrobe_seg.data import split_image_ids


def test_split_is_disjoint_complete_and_deterministic() -> None:
    ids = [f"image-{index}" for index in range(20)]
    train = split_image_ids(ids, "train", val_fraction=0.2, seed=7)
    val = split_image_ids(ids, "val", val_fraction=0.2, seed=7)
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(ids)
    assert val == split_image_ids(list(reversed(ids)), "val", val_fraction=0.2, seed=7)

