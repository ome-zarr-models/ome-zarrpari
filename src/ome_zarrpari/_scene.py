"""
Placing the images of an OME-Zarr 0.6 scene in a scene coordinate system.

The coordinate systems of a scene and its images form a graph, with coordinate
transformations as edges. Each image is shown in a scene coordinate system by
the affine transform along the shortest path from the image's intrinsic
coordinate system to the scene coordinate system.
"""

import numpy as np
import ome_zarr_models.v06
from ome_zarr_models.v06.coordinate_transforms import (
    CoordinateSystemIdentifier,
)

from ome_zarrpari._transforms import TransformGraph

# A coordinate system: (path of the group it is defined in,
# relative to the scene root, coordinate system name)
Node = tuple[str, str]


def _join(*parts: str | None) -> str:
    return "/".join(
        part.strip("/") for part in parts if part and part.strip("/")
    )


def _intrinsic_name(image: ome_zarr_models.v06.Image) -> str:
    return image.ome_attributes.multiscales[0].intrinsic_coordinate_system.name


class SceneGraph(TransformGraph):
    """
    Graph of the coordinate systems in an OME-Zarr 0.6 scene.

    Parameters
    ----------
    scene :
        Scene model.
    """

    def __init__(self, scene: ome_zarr_models.v06.Scene):
        super().__init__()
        self.images = scene.images
        scene_attrs = scene.ome_attributes.scene
        self.coordinate_systems: list[str] = [
            cs.name for cs in scene_attrs.coordinateSystems or ()
        ]

        # Transforms between coordinate systems of each image
        for path, image in self.images.items():
            for multiscale in image.ome_attributes.multiscales:
                for transform in multiscale.coordinateTransformations or ():
                    self.add_transform(
                        transform,
                        self._resolve(transform.input, base_path=path),
                        self._resolve(transform.output, base_path=path),
                    )
        # Transforms between the images and the scene
        for transform in scene_attrs.coordinateTransformations:
            self.add_transform(
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

    def image_to_scene(
        self, image_path: str, coordinate_system: str
    ) -> np.ndarray | None:
        """
        Homogeneous affine matrix from the intrinsic coordinate system of an
        image to a scene coordinate system, or None if there is no
        (affine) path between them.
        """
        source = (image_path, _intrinsic_name(self.images[image_path]))
        return self.find_matrix(source, ("", coordinate_system))
