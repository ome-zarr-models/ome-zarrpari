"""
Fusing multiscale images (e.g. the tiles of a scene) into a single lazy
multiscale image, so napari only needs to render one layer instead of one
per image.

Each image is placed by its homogeneous matrix from (highest resolution)
array indices to world coordinates. Images that are only shifted relative
to the fused pixels are copied (rounded to whole pixels), others (rotated,
scaled) are resampled. Overlapping pixels show the maximum value for images,
and the last non-zero value for labels.

Based on FusedScene in https://github.com/ome/napari-ome-zarr/pull/175.
"""

import itertools
from dataclasses import dataclass

import dask.array as da
import numpy as np
from scipy import ndimage

# Chunk size of the last two axes of the fused image.
# 1024 or 2048 are usually the sweet spots for GPU texture uploads.
CHUNK_SIZE = 1024


@dataclass
class Tile:
    """
    A multiscale image to fuse.
    """

    # Multiscale levels, highest resolution first
    data: list[da.Array]
    # Homogeneous matrix from highest resolution array indices to world
    data_to_world: np.ndarray

    @property
    def level_matrices(self) -> list[np.ndarray]:
        """
        Homogeneous matrices from array indices to world, per level.
        """
        # napari places pixel i of a multiscale level
        # at level 0 pixel i * downsample factor
        level0_shape = np.array(self.data[0].shape)
        return [
            self.data_to_world
            @ np.diag([*(level0_shape / np.array(level.shape)), 1])
            for level in self.data
        ]


@dataclass
class FusedImage:
    """
    A fused multiscale image.
    """

    # Multiscale levels, highest resolution first
    data: list[da.Array]
    # Scale and translation of the highest resolution level
    scale: tuple[float, ...]
    translate: tuple[float, ...]


def _box_corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """All corners of the box [lo, hi], as rows of homogeneous coordinates."""
    corners = np.array(
        list(itertools.product(*zip(lo, hi, strict=True))), dtype=float
    )
    return np.hstack([corners, np.ones((len(corners), 1))])


class _Placement:
    """
    An image level, placed in a fused image level by the matrix from image
    to fused array indices.
    """

    def __init__(self, array: da.Array, matrix: np.ndarray, order: int):
        ndim = len(matrix) - 1
        self.array = array
        self.order = order
        self.shape = np.array(array.shape)
        self.inverse = np.linalg.inv(matrix)
        self.offset: np.ndarray | None
        if np.allclose(matrix[:-1, :-1], np.eye(ndim), atol=1e-6):
            # Only shifted: copy, rounded to whole pixels
            self.offset = np.round(matrix[:-1, -1]).astype(int)
            self.lo, self.hi = self.offset, self.offset + self.shape
        else:
            # Rotated / scaled: resample
            self.offset = None
            corners = (
                _box_corners(-0.5 * np.ones(ndim), self.shape - 0.5) @ matrix.T
            )
            self.lo = np.floor(corners[:, :-1].min(axis=0)).astype(int)
            self.hi = np.ceil(corners[:, :-1].max(axis=0)).astype(int) + 1

    def read_region(
        self, lo: np.ndarray, hi: np.ndarray
    ) -> tuple[da.Array | None, np.ndarray | None]:
        """
        Lazy image data needed for fused pixels [lo, hi),
        with the image index of its first pixel (None if only shifted).
        """
        if self.offset is not None:
            return (
                self.array[
                    tuple(
                        slice(a, b)
                        for a, b in zip(
                            lo - self.offset, hi - self.offset, strict=True
                        )
                    )
                ],
                None,
            )
        corners = _box_corners(lo, hi - 1) @ self.inverse.T
        region_lo = np.clip(
            np.floor(corners[:, :-1].min(axis=0)).astype(int) - 1,
            0,
            self.shape,
        )
        region_hi = np.clip(
            np.ceil(corners[:, :-1].max(axis=0)).astype(int) + 2,
            0,
            self.shape,
        )
        if np.any(region_lo >= region_hi):
            return None, None
        return (
            self.array[
                tuple(
                    slice(a, b)
                    for a, b in zip(region_lo, region_hi, strict=True)
                )
            ],
            region_lo,
        )

    def resample(
        self,
        data: np.ndarray,
        region_lo: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> np.ndarray:
        """Image data at fused pixels [lo, hi) as float, NaN outside the image."""
        linear = self.inverse[:-1, :-1]
        offset = linear @ lo + self.inverse[:-1, -1] - region_lo
        return ndimage.affine_transform(
            # float64 keeps label values exact
            data.astype(np.float64 if self.order == 0 else np.float32),
            linear,
            offset,
            output_shape=tuple(hi - lo),
            order=self.order,
            mode="constant",
            cval=np.nan,
        )


def fuse(tiles: list[Tile], *, is_labels: bool) -> FusedImage:
    """
    Fuse multiscale images with the same number of dimensions and dtype
    into a single lazy multiscale image.

    The fused image has the finest pixel size of the images, and covers all
    images. Overlapping pixels show the maximum value for images, and the
    last non-zero value for labels.
    """
    ndim = tiles[0].data[0].ndim
    dtype = tiles[0].data[0].dtype
    matrices = [tile.level_matrices for tile in tiles]
    # Per tile and level: the pixel size along each world axis
    pixel_sizes = [
        [np.linalg.norm(matrix[:-1, :-1], axis=1) for matrix in tile_matrices]
        for tile_matrices in matrices
    ]

    # Fused image: the finest pixel size, covering all tiles;
    # fused pixel 0 is centered at origin
    pixel_size = np.min([sizes[0] for sizes in pixel_sizes], axis=0)
    corners = np.vstack(
        [
            _box_corners(
                -0.5 * np.ones(ndim), np.array(tile.data[0].shape) - 0.5
            )
            @ tile_matrices[0].T
            for tile, tile_matrices in zip(tiles, matrices, strict=True)
        ]
    )[:, :-1]
    bbox_min, bbox_max = corners.min(axis=0), corners.max(axis=0)
    origin = bbox_min + pixel_size / 2
    level0_shape = np.maximum(
        np.ceil((bbox_max - bbox_min) / pixel_size - 1e-6).astype(int), 1
    )

    # Downsampling factors per level and axis, from the tile with most levels
    reference = max(range(len(tiles)), key=lambda i: len(tiles[i].data))
    level_factors = [
        np.maximum(np.round(sizes / pixel_sizes[reference][0]), 1)
        for sizes in pixel_sizes[reference]
    ]

    pyramid = []
    for factor in level_factors:
        fused_matrix = np.eye(ndim + 1)
        fused_matrix[:-1, :-1] = np.diag(pixel_size * factor)
        fused_matrix[:-1, -1] = origin
        fused_inverse = np.linalg.inv(fused_matrix)

        placements = []
        for tile, tile_matrices, tile_sizes in zip(
            tiles, matrices, pixel_sizes, strict=True
        ):
            # The coarsest tile level that isn't coarser than this fused level
            tile_level = 0
            for index, sizes in enumerate(tile_sizes):
                if np.all(sizes <= pixel_size * factor * (1 + 1e-3)):
                    tile_level = index
            placements.append(
                _Placement(
                    tile.data[tile_level],
                    fused_inverse @ tile_matrices[tile_level],
                    order=0 if is_labels else 1,
                )
            )

        shape = tuple(
            np.maximum(np.ceil(level0_shape / factor).astype(int), 1)
        )
        chunks = (1,) * (ndim - 2) + (CHUNK_SIZE,) * min(ndim, 2)
        template = da.zeros(shape, chunks=chunks, dtype=dtype)
        pyramid.append(
            template.map_blocks(
                _render_chunk,
                dtype=dtype,
                placements=placements,
                is_labels=is_labels,
            )
        )

    return FusedImage(
        data=pyramid,
        scale=tuple(float(s) for s in pixel_size),
        translate=tuple(float(t) for t in origin),
    )


def _render_chunk(
    block: np.ndarray,
    *,
    placements: list[_Placement],
    is_labels: bool,
    block_info: dict | None = None,
) -> np.ndarray:
    """
    Render a chunk of a fused image level from the overlapping images.
    """
    if not block_info:
        return block
    location = block_info[None]["array-location"]
    chunk_lo = np.array([start for start, _ in location])
    chunk_hi = np.array([end for _, end in location])
    chunk_data = np.zeros(block.shape, dtype=block.dtype)

    # 1. Collect the lazy slices of the overlapping images
    tasks = []
    for placement in placements:
        lo = np.maximum(chunk_lo, placement.lo)
        hi = np.minimum(chunk_hi, placement.hi)
        if np.any(lo >= hi):
            continue
        region, region_lo = placement.read_region(lo, hi)
        if region is not None:
            tasks.append((placement, region, region_lo, lo, hi))

    # 2. Fetch all overlapping data in parallel
    fetched = da.compute(*[task[1] for task in tasks])

    # 3. Apply the fetched data to the chunk in order
    for (placement, _, region_lo, lo, hi), values in zip(
        tasks, fetched, strict=True
    ):
        target = chunk_data[
            tuple(
                slice(a, b)
                for a, b in zip(lo - chunk_lo, hi - chunk_lo, strict=True)
            )
        ]
        if region_lo is None:
            valid = (
                values != 0 if is_labels else np.ones(values.shape, dtype=bool)
            )
        else:
            values = placement.resample(values, region_lo, lo, hi)
            valid = ~np.isnan(values)
            if np.issubdtype(block.dtype, np.integer):
                values = np.round(values)
            if is_labels:
                valid &= values != 0
            values = np.where(valid, values, 0).astype(block.dtype)
        if is_labels:
            target[valid] = values[valid]
        else:
            target[valid] = np.maximum(target[valid], values[valid])
    return chunk_data
