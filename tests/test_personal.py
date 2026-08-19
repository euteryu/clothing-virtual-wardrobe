from pathlib import Path

import pytest

from wardrobe_seg.personal import discover_personal_images


def test_discover_personal_images_filters_and_sorts(tmp_path: Path) -> None:
    for name in ("b.JPG", "a.png", "notes.txt", "nested.jpeg"):
        (tmp_path / name).write_bytes(b"test")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "ignored.jpg").write_bytes(b"test")

    assert [path.name for path in discover_personal_images(tmp_path)] == [
        "a.png",
        "b.JPG",
        "nested.jpeg",
    ]


def test_discover_personal_images_requires_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(NotADirectoryError):
        discover_personal_images(missing)
