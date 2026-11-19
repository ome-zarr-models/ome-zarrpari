"""
Placing the images of an OME-Zarr 0.6 scene in a scene coordinate system.

The coordinate systems of a scene and its images form a graph, with coordinate
transformations as edges. Each image is shown in a scene coordinate system by
the affine transform along the shortest path from the image's intrinsic
coordinate system to the scene coordinate system.
"""

import warnings
from collections import deque

import numpy as np
import ome_zarr_models.v06
from ome_zarr_models.v06.coordinate_transforms import (
    AnyTransform,
    CoordinateSystemIdentifier,
)

# A coordinate system: (path of the group it is defined in,
# relative to the scene root, coordinate system name)
Node = tuple[str, str]


def _join(*parts: str | None) -> str:
    return "/".join(
        part.strip("/") for part in parts if part and part.strip("/")
    )


def _homogeneous_matrix(transform: AnyTransform) -> np.ndarray | None:
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


def _intrinsic_name(image: ome_zarr_models.v06.Image) -> str:
    return image.ome_attributes.multiscales[0].intrinsic_coordinate_system.name


class SceneGraph:
    """
    Graph of the coordinate systems in an OME-Zarr 0.6 scene.

    Parameters
    ----------
    scene :
        Scene model.
    """

    def __init__(self, scene: ome_zarr_models.v06.Scene):
        self.images = scene.images
        scene_attrs = scene.ome_attributes.scene
        self.coordinate_systems: list[str] = [
            cs.name for cs in scene_attrs.coordinateSystems or ()
        ]
        self._edges: dict[Node, list[tuple[Node, np.ndarray]]] = {}

        # Transforms between coordinate systems of each image
        for path, image in self.images.items():
            for multiscale in image.ome_attributes.multiscales:
                for transform in multiscale.coordinateTransformations or ():
                    self._add_transform(
                        transform,
                        self._resolve(transform.input, base_path=path),
                        self._resolve(transform.output, base_path=path),
                    )
        # Transforms between the images and the scene
        for transform in scene_attrs.coordinateTransformations:
            self._add_transform(
                transform,
                self._resolve(transform.input),
                self._resolve(transform.output),
            )

    def _resolve(
        self,
        identifier: CoordinateSystemIdentifier | None,
        *,
        base_path: str = "",
    ) -> Node | None:
        """
        Get the graph node of a coordinate system identifier.
        """
        if identifier is None:
            return None
        path = _join(base_path, identifier.path)
        if identifier.name is not None:
            return (path, identifier.name)
        # No name given: use the intrinsic coordinate system of the image
        if path in self.images:
            return (path, _intrinsic_name(self.images[path]))
        return None

    def _add_transform(
        self, transform: AnyTransform, source: Node | None, target: Node | None
    ) -> None:
        if source is None or target is None:
            warnings.warn(
                f"Ignoring '{transform.type}' transform with unresolved "
                f"input/output ({transform.input} -> {transform.output})",
                stacklevel=3,
            )
            return
        matrix = _homogeneous_matrix(transform)
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

    def _find_matrix(self, source: Node, target: Node) -> np.ndarray | None:
        """
        Homogeneous affine matrix of the shortest path from source to target,
        or None if there is no path. Source and target must differ.
        """
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

    def image_to_scene(
        self, image_path: str, coordinate_system: str
    ) -> np.ndarray | None:
        """
        Homogeneous affine matrix from the intrinsic coordinate system of an
        image to a scene coordinate system, or None if there is no
        (affine) path between them.
        """
        source = (image_path, _intrinsic_name(self.images[image_path]))
        return self._find_matrix(source, ("", coordinate_system))


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
