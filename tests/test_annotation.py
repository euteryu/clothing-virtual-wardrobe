import numpy as np
from pycocotools import mask as mask_utils

from wardrobe_seg.annotation import encode_coco_mask


def test_coco_mask_round_trip_is_json_safe() -> None:
    mask = np.zeros((8, 6), dtype=np.uint8)
    mask[2:7, 1:5] = 1
    encoded = encode_coco_mask(mask)
    assert isinstance(encoded["counts"], str)
    decoded = mask_utils.decode(
        {"size": encoded["size"], "counts": encoded["counts"].encode("ascii")}
    )
    np.testing.assert_array_equal(decoded, mask)
