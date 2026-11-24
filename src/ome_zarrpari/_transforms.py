"""
Graphs of coordinate systems connected by OME-Zarr 0.6 coordinate
transformations, and conversion of their affine matrices to napari affines.
"""

import warnings
from collections import deque
from collections.abc import Hashable

import numpy as np
from ome_zarr_models.v06.coordinate_transforms import AnyTransform

# A coordinate system
Node = Hashable


def homogeneous_matrix(transform: AnyTransform) -> np.ndarray | None:
    """
    Homogeneous affine matrix of a transform, with shape
    (output ndim + 1, input ndim + 1), or None if the transform
    can't be converted to an affine.
    """
    try:
        affine = np.asarray(transform.as_affine().affine_matrix, dtype=float)
    except (RuntimeError, ValueError):
        # NoAffineError and NotImplementedError are RuntimeErrors
        return None
    n_out, n_in = affine.shape[0], affine.shape[1] - 1
    matrix = np.zeros((n_out + 1, n_in + 1))
    matrix[:n_out] = affine
    matrix[n_out, n_in] = 1
    return matrix


class TransformGraph:
    """
    Graph of coordinate systems (nodes), with affine coordinate
    transformations between them as edges.
    """

    def __init__(self) -> None:
        self._edges: dict[Node, list[tuple[Node, np.ndarray]]] = {}

    def add_transform(
        self, transform: AnyTransform, source: Node | None, target: Node | None
    ) -> None:
        """
        Add a transform from source to target, and its inverse if it has one.

        Transforms with an unknown source or target, or that can't be
        converted to an affine transform, are ignored with a warning.
        """
        if source is None or target is None:
            warnings.warn(
                f"Ignoring '{transform.type}' transform with unresolved "
                f"input/output ({transform.input} -> {transform.output})",
                stacklevel=3,
            )
            return
        matrix = homogeneous_matrix(transform)
        if matrix is None:
            warnings.warn(
                f"Ignoring '{transform.type}' transform from {source} to "
                f"{target}: it can't be converted to an affine transform",
                stacklevel=3,
            )
            return
        self._edges.setdefault(source, []).append((target, matrix))
        # Also allow going the other way, if the transform is invertible
        if matrix.shape[0] == matrix.shape[1]:
            try:
                inverse = np.linalg.inv(matrix)
            except np.linalg.LinAlgError:
                return
            self._edges.setdefault(target, []).append((source, inverse))

    def find_matrix(self, source: Node, target: Node) -> np.ndarray | None:
        """
        Homogeneous affine matrix of the shortest path from source to target,
        or None if there is no path.
        """
        if source == target:
            return None
        found: dict[Node, np.ndarray] = {}
        queue: deque[Node] = deque([source])
        while queue:
            node = queue.popleft()
            for next_node, edge_matrix in self._edges.get(node, []):
                if next_node == source or next_node in found:
                    continue
                if node == source:
                    matrix = edge_matrix
                elif found[node].shape[0] == edge_matrix.shape[1]:
                    matrix = edge_matrix @ found[node]
                else:
                    # Dimensionality mismatch
                    continue
                if next_node == target:
                    return matrix
                found[next_node] = matrix
                queue.append(next_node)
        return None


def to_napari_affine(
    matrix: np.ndarray, channel_axis: int | None
) -> np.ndarray | None:
    """
    Convert a homogeneous affine matrix to a napari layer affine, removing
    the channel axis if present.

    Returns None if this isn't possible: napari needs the same number of
    input and output dimensions, and the channel axis must be left unchanged.
    """
    if matrix.shape[0] != matrix.shape[1]:
        return None
    if channel_axis is not None:
        row = np.zeros(matrix.shape[1])
        row[channel_axis] = 1
        column = np.zeros(matrix.shape[0])
        column[channel_axis] = 1
        if not (
            np.allclose(matrix[channel_axis], row)
            and np.allclose(matrix[:, channel_axis], column)
        ):
            return None
        matrix = np.delete(
            np.delete(matrix, channel_axis, axis=0), channel_axis, axis=1
        )
    return matrix
