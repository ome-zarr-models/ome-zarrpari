"""
Tests for fusing multiscale images into a single lazy multiscale image.
"""

import dask.array as da
import numpy as np
import pytest

import ome_zarrpari._fusion
from ome_zarrpari._fusion import Tile, fuse


def _shift(translation: tuple[float, ...], scale=(1.0, 1.0)) -> np.ndarray:
    matrix = np.eye(len(translation) + 1)
    matrix[:-1, :-1] = np.diag(scale)
    matrix[:-1, -1] = translation
    return matrix


def _multiscale(array: np.ndarray, n_levels: int = 1) -> list[da.Array]:
    return [
        da.from_array(array[(slice(None, None, 2**level),) * array.ndim])
        for level in range(n_levels)
    ]


def test_fuse_shifted() -> None:
    """
    Shifted images are copied to their place in the fused image.
    """
    rng = np.random.default_rng(0)
    a = rng.integers(1, 100, size=(4, 6), dtype=np.uint16)
    b = rng.integers(1, 100, size=(4, 6), dtype=np.uint16)
    fused = fuse(
        [
            Tile(_multiscale(a), _shift((10, 20), scale=(0.5, 2))),
            Tile(_multiscale(b), _shift((13, 32), scale=(0.5, 2))),
        ],
        is_labels=False,
    )
    assert fused.scale == (0.5, 2)
    assert fused.translate == (10, 20)
    (level,) = fused.data
    assert level.dtype == np.uint16
    result = level.compute()
    assert result.shape == (10, 12)
    np.testing.assert_array_equal(result[:4, :6], a)
    np.testing.assert_array_equal(result[6:, 6:], b)
    # Not covered by any image
    assert np.all(result[4:6] == 0)
    assert np.all(result[:4, 6:] == 0)


@pytest.mark.parametrize("is_labels", [False, True])
def test_fuse_overlap(is_labels) -> None:
    """
    Overlapping pixels show the maximum value for images, and the last
    non-zero value for labels.
    """
    a = np.array([[5, 1, 0]], dtype=np.uint8)
    b = np.array([[2, 0, 3]], dtype=np.uint8)
    fused = fuse(
        [
            Tile(_multiscale(a), _shift((0, 0))),
            Tile(_multiscale(b), _shift((0, 0))),
        ],
        is_labels=is_labels,
    )
    expected = [[2, 1, 3]] if is_labels else [[5, 1, 3]]
    np.testing.assert_array_equal(fused.data[0].compute(), expected)


def test_fuse_rotated_labels() -> None:
    """
    Rotated images are resampled; labels with nearest neighbour interpolation.
    """
    labels = np.arange(1, 13, dtype=np.uint32).reshape(3, 4)
    # Rotate by 90 degrees: (y, x) -> (-x, y)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    fused = fuse(
        [Tile(_multiscale(labels), rotation)],
        is_labels=True,
    )
    assert fused.scale == (1, 1)
    assert fused.translate == (-3, 0)
    np.testing.assert_array_equal(fused.data[0].compute(), np.rot90(labels))


def test_fuse_multiscale() -> None:
    """
    Each fused level is made from the matching image levels.
    """
    rng = np.random.default_rng(0)
    a = rng.integers(1, 100, size=(8, 8), dtype=np.uint8)
    b = rng.integers(1, 100, size=(8, 8), dtype=np.uint8)
    fused = fuse(
        [
            Tile(_multiscale(a, n_levels=2), _shift((0, 0))),
            Tile(_multiscale(b, n_levels=2), _shift((0, 8))),
        ],
        is_labels=False,
    )
    assert [level.shape for level in fused.data] == [(8, 16), (4, 8)]
    np.testing.assert_array_equal(
        fused.data[1].compute(), np.hstack([a[::2, ::2], b[::2, ::2]])
    )


def test_fuse_chunks(monkeypatch) -> None:
    """
    Fused images are rendered per chunk, from the images overlapping each
    chunk. Chunks outside a rotated image stay empty.
    """
    monkeypatch.setattr(ome_zarrpari._fusion, "CHUNK_SIZE", 2)
    labels = np.ones((8, 8), dtype=np.uint8)
    angle = np.pi / 4
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    shifted = np.full((2, 2), 2, dtype=np.uint8)
    fused = fuse(
        [
            Tile(_multiscale(labels), rotation),
            Tile(_multiscale(shifted), _shift((20, 20))),
        ],
        is_labels=True,
    )
    (level,) = fused.data
    assert level.chunksize == (2, 2)
    result = level.compute()
    # The rotated square is a diamond: its bounding box corners are empty
    assert result[0, 0] == 0
    assert result[0, -1] == 0
    centre = np.round(
        (rotation[:2, :2] @ (3.5, 3.5) - fused.translate) / fused.scale
    ).astype(int)
    assert result[tuple(centre)] == 1
    # The shifted image is copied, rounded to whole fused pixels
    y, x = np.round(
        (np.array((20, 20)) - fused.translate) / fused.scale
    ).astype(int)
    np.testing.assert_array_equal(result[y : y + 2, x : x + 2], shifted)
