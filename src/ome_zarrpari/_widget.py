import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import dask.array as da
import napari.layers
import numpy as np
import ome_zarr_models.v04
import ome_zarr_models.v04.multiscales
import ome_zarr_models.v05
import ome_zarr_models.v05.multiscales
import ome_zarr_models.v06
import ome_zarr_models.v06.multiscales
import zarr
from napari.settings import get_settings
from napari.utils.colormaps import CMYBGR, MAGENTA_GREEN
from ome_zarr_models import open_ome_zarr
from ome_zarr_models.common.coordinate_transformations import (
    VectorScale,
    VectorTranslation,
)
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ome_zarrpari._fusion import Tile, fuse
from ome_zarrpari._scene import SceneGraph
from ome_zarrpari._transforms import TransformGraph, to_napari_affine

if TYPE_CHECKING:
    import napari

from qtpy.QtWidgets import QLabel

AnyImage = (
    ome_zarr_models.v04.Image
    | ome_zarr_models.v05.Image
    | ome_zarr_models.v06.Image
)
AnyImageLabel = (
    ome_zarr_models.v04.ImageLabel
    | ome_zarr_models.v05.ImageLabel
    | ome_zarr_models.v06.ImageLabel
)

SUPPORTED_CLASSES = (AnyImage, AnyImageLabel, ome_zarr_models.v06.Scene)

AnyMultiscale = (
    ome_zarr_models.v04.multiscales.Multiscale
    | ome_zarr_models.v05.multiscales.Multiscale
    | ome_zarr_models.v06.multiscales.Multiscale
)

AnyAxis = (
    ome_zarr_models.v04.axes.Axis
    | ome_zarr_models.v05.axes.Axis
    | ome_zarr_models.v06.coordinate_transforms.Axis
)

IMAGE_LABEL_CLASSES: dict[str, type[AnyImageLabel]] = {
    "0.4": ome_zarr_models.v04.ImageLabel,
    "0.5": ome_zarr_models.v05.ImageLabel,
    "0.6": ome_zarr_models.v06.ImageLabel,
}

AddedLayers = dict[
    napari.layers.Image | napari.layers.Labels, AnyImage | AnyImageLabel
]


# Layers fused together: (layer type, label image path, channel index,
# number of dimensions, dtype)
_FuseGroup = tuple[
    Literal["image", "labels"], str | None, int | None, int, np.dtype
]


@dataclass
class LayerSpec:
    """
    A napari layer, before it is added to the viewer.
    """

    # Multiscale levels, highest resolution first
    data: list[da.Array]
    layer_type: Literal["image", "labels"]
    # Keyword arguments for napari's add_image/add_labels
    kwargs: dict[str, Any]
    # The OME-Zarr model the data is from
    model: AnyImage | AnyImageLabel
    # The path of the label image within the labels group, for labels
    label: str | None = None
    # The multiscale the layer is made from (None for fused layers)
    multiscale: AnyMultiscale | None = None

    @property
    def channel_axis(self) -> int | None:
        return self.kwargs.get("channel_axis")

    @property
    def ndim(self) -> int:
        """
        Number of layer dimensions (the channel axis isn't a layer dimension).
        """
        ndim = self.data[0].ndim
        return ndim if self.channel_axis is None else ndim - 1

    def add_to(self, viewer: "napari.Viewer") -> AddedLayers:
        """
        Add this layer (one per channel for images) to a viewer.
        """
        if self.layer_type == "image":
            layers = viewer.add_image(
                self.data, multiscale=True, **self.kwargs
            )
        else:
            layers = viewer.add_labels(
                self.data, multiscale=True, **self.kwargs
            )
        if not isinstance(layers, list):
            layers = [layers]
        return dict.fromkeys(layers, self.model)


def _add_layers(
    viewer: "napari.Viewer", specs: list[LayerSpec]
) -> AddedLayers:
    added_layers: AddedLayers = {}
    for spec in specs:
        added_layers.update(spec.add_to(viewer))
    return added_layers


class LoadedScene:
    """
    The layers of an OME-Zarr 0.6 scene, shown in a napari viewer.

    The scene images are either shown as separate layers (one or more per
    scene image), or fused into one layer per channel and label image.

    Parameters
    ----------
    viewer :
        Viewer to show the scene in.
    name :
        Name of the scene, used for the names of fused layers.
    graph :
        Coordinate system graph of the scene.
    specs :
        Scene image path -> layers of that image.
    """

    def __init__(
        self,
        viewer: "napari.Viewer",
        name: str,
        graph: SceneGraph,
        specs: dict[str, list[LayerSpec]],
    ):
        self.viewer = viewer
        self.name = name
        self.graph = graph
        self.specs = specs
        # Layers currently shown
        self.layers: AddedLayers = {}
        # Whether the layers are fused, None if not shown yet
        self.fused: bool | None = None
        # Separate layers per scene image path, when not fused
        self._separate_layers: dict[str, list[napari.layers.Layer]] = {}

    @property
    def coordinate_systems(self) -> list[str]:
        return self.graph.coordinate_systems

    def show(
        self, coordinate_system: str | None, *, fused: bool
    ) -> tuple[list[napari.layers.Layer], AddedLayers]:
        """
        Show the scene in a scene coordinate system (or None to show each
        image in its intrinsic coordinate system), replacing any layers
        already shown.

        Layers that can't be placed in the scene coordinate system are shown
        as separate layers in the intrinsic coordinate system of their image,
        with a warning.

        Returns
        -------
        removed :
            Layers removed from the viewer.
        added :
            Layers added to the viewer.
        """
        affines = {
            path: self._get_affine(path, coordinate_system)
            for path in self.specs
        }
        if self.fused is False and not fused:
            # Already showing separate layers: only update their affines
            removed = self._forget_removed_layers()
            not_placed = []
            for path, layers in self._separate_layers.items():
                for layer in layers:
                    affine = _fit_affine(affines[path], layer.ndim)
                    if affine is None:
                        not_placed.append(layer.name)
                        affine = np.eye(layer.ndim + 1)
                    layer.affine = affine
            self._warn_not_placed(not_placed, coordinate_system)
            return removed, {}

        removed = self._remove_layers()
        self._separate_layers = {}
        if fused:
            added = self._add_fused(affines, coordinate_system)
        else:
            added = self._add_separate(affines, coordinate_system)
        self.layers = dict(added)
        self.fused = fused
        return removed, added

    def _get_affine(
        self, path: str, coordinate_system: str | None
    ) -> np.ndarray | None:
        """
        napari affine (without channel axis) from the intrinsic coordinate
        system of a scene image to a scene coordinate system.
        """
        if coordinate_system is None:
            return None
        matrix = self.graph.image_to_scene(path, coordinate_system)
        if matrix is None:
            return None
        # The channel axis isn't a layer dimension, for both the image
        # layers (one per channel) and the label layers (no channels)
        channel_axis = _get_channel_axis(
            self.graph.images[path].ome_attributes.multiscales[0]
        )
        return to_napari_affine(matrix, channel_axis)

    def _add_separate(
        self,
        affines: dict[str, np.ndarray | None],
        coordinate_system: str | None,
    ) -> AddedLayers:
        added: AddedLayers = {}
        not_placed = []
        for path, specs in self.specs.items():
            layers = _add_layers(self.viewer, specs)
            added.update(layers)
            self._separate_layers[path] = list(layers)
            for layer in layers:
                affine = _fit_affine(affines[path], layer.ndim)
                if affine is None:
                    not_placed.append(layer.name)
                else:
                    layer.affine = affine
        self._warn_not_placed(not_placed, coordinate_system)
        return added

    def _add_fused(
        self,
        affines: dict[str, np.ndarray | None],
        coordinate_system: str | None,
    ) -> AddedLayers:
        # Group the (channels of the) layers that are fused together
        groups: dict[
            _FuseGroup, list[tuple[LayerSpec, int | None, np.ndarray]]
        ] = {}
        not_placed_specs: list[LayerSpec] = []
        for path, specs in self.specs.items():
            for spec in specs:
                affine: np.ndarray | None
                if coordinate_system is None:
                    affine = np.eye(spec.ndim + 1)
                else:
                    affine = _fit_affine(affines[path], spec.ndim)
                if affine is None:
                    not_placed_specs.append(spec)
                    continue
                channels: list[int | None] = (
                    [None]
                    if spec.channel_axis is None
                    else list(range(spec.data[0].shape[spec.channel_axis]))
                )
                for channel in channels:
                    key: _FuseGroup = (
                        spec.layer_type,
                        spec.label,
                        channel,
                        spec.ndim,
                        spec.data[0].dtype,
                    )
                    groups.setdefault(key, []).append((spec, channel, affine))

        n_channels = len(
            {key[2] for key in groups if key[0] == "image"} - {None}
        )
        dtypes_per_group: dict[tuple, set[np.dtype]] = {}
        for group in groups:
            dtypes_per_group.setdefault(group[:4], set()).add(group[4])

        added: AddedLayers = {}
        for group, members in groups.items():
            layer_type, label, channel, _, dtype = group
            tiles = [
                Tile(
                    data=_select_channel(spec, channel),
                    data_to_world=affine @ _scale_translate_matrix(spec),
                )
                for spec, _, affine in members
            ]
            fused = fuse(tiles, is_labels=layer_type == "labels")

            name = self.name
            if label is not None:
                name = f"{name}: {label}"
            elif channel is not None and n_channels > 1:
                name = f"{name} [{channel}]"
            if len(dtypes_per_group[group[:4]]) > 1:
                name = f"{name} ({dtype})"

            first_spec = members[0][0]
            kwargs = {
                k: v
                for k, v in first_spec.kwargs.items()
                if k not in ("channel_axis", "name", "scale", "translate")
            }
            kwargs.update(
                name=name, scale=fused.scale, translate=fused.translate
            )
            if layer_type == "image" and channel is not None:
                kwargs.update(_channel_display(channel, n_channels))
            spec = LayerSpec(
                data=fused.data,
                layer_type=layer_type,
                kwargs=kwargs,
                model=first_spec.model,
                label=label,
            )
            added.update(spec.add_to(self.viewer))

        # Layers that can't be placed are shown as separate layers
        not_placed = _add_layers(self.viewer, not_placed_specs)
        added.update(not_placed)
        self._warn_not_placed(
            [layer.name for layer in not_placed], coordinate_system
        )
        return added

    def _forget_removed_layers(self) -> list[napari.layers.Layer]:
        """
        Forget about layers that were removed from the viewer.
        """
        removed = [
            layer for layer in self.layers if layer not in self.viewer.layers
        ]
        for layer in removed:
            del self.layers[layer]
        for path, layers in self._separate_layers.items():
            self._separate_layers[path] = [
                layer for layer in layers if layer in self.viewer.layers
            ]
        return removed

    def _remove_layers(self) -> list[napari.layers.Layer]:
        """
        Remove the shown layers from the viewer.
        """
        removed = list(self.layers)
        for layer in removed:
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        self.layers = {}
        return removed

    @staticmethod
    def _warn_not_placed(
        layer_names: list[str], coordinate_system: str | None
    ) -> None:
        if layer_names and coordinate_system is not None:
            warnings.warn(
                f"Could not place layers {layer_names} in scene coordinate "
                f"system '{coordinate_system}' with an affine transform. "
                "Showing them in their intrinsic coordinate system instead.",
                stacklevel=3,
            )


def _fit_affine(affine: np.ndarray | None, ndim: int) -> np.ndarray | None:
    """
    The affine if it fits a layer with ndim dimensions, otherwise None.

    Labels are placed like their image, which assumes they have the same
    (non-channel) dimensions.
    """
    if affine is None or len(affine) != ndim + 1:
        return None
    return affine


def _select_channel(spec: LayerSpec, channel: int | None) -> list[da.Array]:
    """
    The multiscale levels of one channel of a layer.
    """
    if channel is None or spec.channel_axis is None:
        return spec.data
    index = (slice(None),) * spec.channel_axis + (channel,)
    return [level[index] for level in spec.data]


def _scale_translate_matrix(spec: LayerSpec) -> np.ndarray:
    """
    Homogeneous matrix of the scale and translation of a layer.
    """
    ndim = spec.ndim
    matrix = np.eye(ndim + 1)
    matrix[:ndim, :ndim] = np.diag(spec.kwargs.get("scale", np.ones(ndim)))
    matrix[:ndim, ndim] = spec.kwargs.get("translate", np.zeros(ndim))
    return matrix


def _channel_display(channel: int, n_channels: int) -> dict[str, str]:
    """
    Colormap and blending of a channel layer, as napari sets them when
    adding an image with a channel axis.
    """
    if n_channels == 1:
        colormap = "gray"
    elif n_channels == 2:
        colormap = MAGENTA_GREEN[channel % 2]
    else:
        colormap = CMYBGR[channel % len(CMYBGR)]
    blending = "translucent_no_depth" if channel == 0 else "additive"
    return {"colormap": colormap, "blending": blending}


class OMEZarrpariWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        # Keep track of layers added by this widget
        self.added_layers: AddedLayers = {}
        # Multiscale image each added (non-scene) layer was created from
        self._layer_multiscales: dict[napari.layers.Layer, AnyMultiscale] = {}
        # Coordinate system each added layer is shown in, if not intrinsic
        self._layer_coordinate_systems: dict[napari.layers.Layer, str] = {}
        # The most recently loaded scene
        self.scene: LoadedScene | None = None

        # Create text box and button
        self.text_box = QLineEdit()
        self.text_box.setPlaceholderText("Enter OME-Zarr path or URL...")

        # Browse button
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self._on_browse)

        # Fuse the images of scenes into one layer per channel/label image
        self.fuse_checkbox = QCheckBox("Load fused scene")
        self.fuse_checkbox.setChecked(False)
        self.fuse_checkbox.setToolTip(
            "Load the images of an OME-Zarr 0.6 scene (e.g. tiles) as a "
            "single layer per channel and label image, instead of separate "
            "layers per image. This is much faster for scenes with many "
            "images. Applies when loading."
        )

        # Only enabled when a path is entered
        self.load_btn = QPushButton("Load OME-Zarr")
        self.load_btn.setEnabled(False)
        self.load_btn.clicked.connect(self._on_load)
        self.text_box.textChanged.connect(
            lambda text: self.load_btn.setEnabled(bool(text.strip()))
        )

        # Create label for status messages
        self.status_text = QLabel("")
        self.status_text.setWordWrap(True)

        # Add widgets to vertical layout
        layout = QVBoxLayout()

        text_row = QHBoxLayout()
        text_row.addWidget(self.text_box)
        text_row.addWidget(self.browse_btn)
        layout.addLayout(text_row)

        layout.addWidget(self.fuse_checkbox)
        layout.addWidget(self.load_btn)
        layout.addWidget(self.status_text)

        # Coordinate system of the selected image (OME-Zarr 0.6)
        layout.addWidget(QLabel("Image coordinate system"))
        self.image_coord_system_dropdown = QComboBox()
        self.image_coord_system_dropdown.setEnabled(False)
        self.image_coord_system_dropdown.setToolTip(
            "Coordinate system to show the selected OME-Zarr 0.6 image in. "
            "Applies to all layers (channels) of the image."
        )
        self.image_coord_system_dropdown.currentTextChanged.connect(
            self._on_image_coord_system_changed
        )
        layout.addWidget(self.image_coord_system_dropdown)
        self.viewer.layers.selection.events.changed.connect(
            self._on_layer_selection_changed
        )

        # Coordinate system of the most recently loaded scene
        layout.addWidget(QLabel("Scene coordinate system"))
        self.coord_system_dropdown = QComboBox()
        self.coord_system_dropdown.setEnabled(False)
        self.coord_system_dropdown.currentTextChanged.connect(
            self._on_coord_system_changed
        )
        layout.addWidget(self.coord_system_dropdown)

        layout.addStretch(100)  # Push everything to the top

        self.napari_async_button = QPushButton("Enable napari async")
        self.napari_async_button.clicked.connect(self._enable_napari_async)
        if self._async_state:
            self.napari_async_button.setEnabled(False)
            self.napari_async_button.setText("napari async enabled")

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self.napari_async_button)
        layout.addLayout(bottom_row)

        self.setLayout(layout)

    @property
    def load_pane_status_text(self) -> str:
        return self.status_text.text()

    def _on_load(self) -> None:
        path = self.text_box.text().strip()
        if path:
            self._clear()
            self._load_ome_zarr(path)

    def _clear(self) -> None:
        """
        Remove the layers loaded by this widget from the viewer, and forget
        any loaded images and scene. Other layers are kept.
        """
        for layer in list(self.added_layers):
            # Skip layers already removed by the user
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        self.added_layers.clear()
        self._layer_multiscales.clear()
        self._layer_coordinate_systems.clear()
        self.scene = None
        self.coord_system_dropdown.blockSignals(True)
        self.coord_system_dropdown.clear()
        self.coord_system_dropdown.setEnabled(False)
        self.coord_system_dropdown.blockSignals(False)
        self._on_layer_selection_changed()

    def _on_browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.text_box.setText(folder)

    @property
    def _async_state(self) -> bool:
        settings = get_settings()
        return settings.experimental.async_

    @_async_state.setter
    def _async_state(self, value: bool) -> None:
        settings = get_settings()
        settings.experimental.async_ = value

    def _enable_napari_async(self) -> None:
        self._async_state = True
        self.napari_async_button.setEnabled(False)
        self.napari_async_button.setText("napari async enabled")

    def _load_ome_zarr(self, path: str, *, visible: bool = True) -> None:
        """
        Load an OME-Zarr file into napari.

        Parameters
        ----------
        path : str
            Path to OME-Zarr group.
        visible : bool
            Set visible status of any created napari layers.
        """
        self.status_text.setText("Loading OME-Zarr...")
        try:
            group = zarr.open_group(path, mode="r")
            data = open_ome_zarr(path)
        except Exception as e:  # noqa: BLE001
            self.status_text.setText(
                "Loading OME-Zarr failed. See console for more details"
            )
            print(f"Error loading OME-Zarr from {path}:")
            print(f"{type(e).__name__}: {str(e)}")
            return

        if not isinstance(data, SUPPORTED_CLASSES):
            self.status_text.setText(
                f"Found {type(data)} - loading not currently supported."
            )
            return

        if isinstance(data, ome_zarr_models.v06.Scene):
            self._set_scene(
                _read_ome_zarr_scene(self.viewer, group, data, visible=visible)
            )
            self._show_scene(fused=self.fuse_checkbox.isChecked())
        else:
            added_layers = _load_ome_zarr_image(
                self.viewer,
                group,
                data,
                visible=visible,
                layer_multiscales=self._layer_multiscales,
            )
            self.added_layers.update(added_layers)
        self.status_text.setText("Successfully loaded")
        self._on_layer_selection_changed()

    def _image_layers_of_selection(self) -> list[napari.layers.Layer]:
        """
        All layers of the multiscale image of the selected layer, if a single
        layer added by this widget is selected.
        """
        selected = list(self.viewer.layers.selection)
        if len(selected) != 1 or selected[0] not in self._layer_multiscales:
            return []
        multiscale = self._layer_multiscales[selected[0]]
        return [
            layer
            for layer, layer_multiscale in self._layer_multiscales.items()
            if layer_multiscale is multiscale and layer in self.viewer.layers
        ]

    def _on_layer_selection_changed(self, event=None) -> None:
        """
        Show the coordinate systems of the selected image in the dropdown.
        """
        layers = self._image_layers_of_selection()
        names = (
            _get_coordinate_systems(self._layer_multiscales[layers[0]])
            if layers
            else []
        )
        dropdown = self.image_coord_system_dropdown
        dropdown.blockSignals(True)
        dropdown.clear()
        dropdown.addItems(names)
        if names:
            dropdown.setCurrentText(
                self._layer_coordinate_systems.get(layers[0], names[0])
            )
        dropdown.setEnabled(len(names) > 1)
        dropdown.blockSignals(False)

    def _on_image_coord_system_changed(self, coordinate_system: str) -> None:
        layers = self._image_layers_of_selection()
        if coordinate_system and layers:
            self._set_coordinate_system(layers, coordinate_system)

    def _set_coordinate_system(
        self, layers: list[napari.layers.Layer], coordinate_system: str
    ) -> None:
        """
        Show the layers of a multiscale image in one of its coordinate
        systems.
        """
        multiscale = self._layer_multiscales[layers[0]]
        placements = [
            _get_coordinate_system_placement(
                multiscale,
                coordinate_system,
                channel_axis=(
                    _get_channel_axis(multiscale)
                    if isinstance(layer, napari.layers.Image)
                    else None
                ),
            )
            for layer in layers
        ]
        valid_placements = [p for p in placements if p is not None]
        if len(valid_placements) != len(placements):
            self.status_text.setText(
                f"Can't show '{multiscale.name}' in coordinate system "
                f"'{coordinate_system}': it has no affine transform from the "
                "intrinsic coordinate system that leaves the channel axis "
                "unchanged."
            )
            self._on_layer_selection_changed()
            return
        for layer, (affine, axis_labels, axis_units) in zip(
            layers, valid_placements, strict=True
        ):
            layer.affine = affine
            if axis_labels is not None:
                layer.axis_labels = axis_labels
            layer.units = axis_units
            self._layer_coordinate_systems[layer] = coordinate_system
        self.status_text.setText(
            f"Showing '{multiscale.name}' in coordinate system "
            f"'{coordinate_system}'"
        )

    def _set_scene(self, scene: LoadedScene) -> None:
        """
        Set the scene that the coordinate system dropdown applies to.
        """
        self.scene = None
        self.coord_system_dropdown.clear()
        self.coord_system_dropdown.addItems(scene.coordinate_systems)
        self.coord_system_dropdown.setEnabled(
            len(scene.coordinate_systems) > 1
        )
        self.scene = scene

    def _show_scene(self, *, fused: bool | None = None) -> None:
        """
        Show the current scene in the selected coordinate system, fused or
        not (by default, as the scene is currently shown).
        """
        if self.scene is None:
            return
        if fused is None:
            fused = bool(self.scene.fused)
        coordinate_system = self.coord_system_dropdown.currentText() or None
        removed, added = self.scene.show(coordinate_system, fused=fused)
        for layer in removed:
            self.added_layers.pop(layer, None)
        self.added_layers.update(added)

    def _on_coord_system_changed(self, coordinate_system: str) -> None:
        if coordinate_system:
            self._show_scene()


def _get_axis_names(multiscale: AnyMultiscale) -> list[str] | None:
    """
    Get axis labels from Multiscale metadata.
    """
    axis_labels_raw = [axis.name for axis in _get_axes(multiscale)]
    if any(label is None for label in axis_labels_raw):
        print(
            f"Warning: At least one axis label is None for multiscale '{multiscale.name}', "
            "not setting any axis labels."
        )
        return None
    else:
        return [str(label) for label in axis_labels_raw]


def _get_axis_units(multiscale: AnyMultiscale) -> list[str]:
    """
    Get axis units from Multiscale metadata.

    Axes without a unit (e.g. channel axes) are given the napari default unit
    ('pixel'), as in napari-ome-zarr.

    # TODO: convert strings to pint units if they make sense as physical units
    """
    return [
        "pixel" if axis.unit is None else str(axis.unit)
        for axis in _get_axes(multiscale)
    ]


def _get_axes(multiscale: AnyMultiscale) -> Sequence[AnyAxis]:
    """
    Get the axes of the (intrinsic) physical coordinate system of a multiscale.
    """
    if isinstance(multiscale, ome_zarr_models.v06.multiscales.Multiscale):
        return multiscale.intrinsic_coordinate_system.axes
    return multiscale.axes


def _coordinate_system_graph(
    multiscale: ome_zarr_models.v06.multiscales.Multiscale,
) -> TransformGraph:
    """
    Graph of the coordinate systems of a multiscale image, connected by its
    coordinate transformations.

    Transformations to coordinate systems of other images (with an output
    path) are not included.
    """
    graph: TransformGraph = TransformGraph()
    for transform in multiscale.coordinateTransformations or ():
        if transform.output is not None and transform.output.path:
            continue
        graph.add_transform(
            transform,
            transform.input.name if transform.input else None,
            transform.output.name if transform.output else None,
        )
    return graph


def _get_coordinate_systems(multiscale: AnyMultiscale) -> list[str]:
    """
    Names of the coordinate systems a multiscale image can be shown in,
    starting with its intrinsic coordinate system.

    Only OME-Zarr 0.6 images have multiple coordinate systems; for earlier
    versions this is empty.
    """
    if not isinstance(multiscale, ome_zarr_models.v06.multiscales.Multiscale):
        return []
    intrinsic = multiscale.intrinsic_coordinate_system.name
    graph = _coordinate_system_graph(multiscale)
    return [intrinsic] + [
        cs.name
        for cs in multiscale.coordinateSystems
        if cs.name != intrinsic
        and graph.find_matrix(intrinsic, cs.name) is not None
    ]


def _get_coordinate_system_placement(
    multiscale: AnyMultiscale,
    coordinate_system: str,
    *,
    channel_axis: int | None,
) -> tuple[np.ndarray, list[str] | None, list[str]] | None:
    """
    Napari affine, axis labels and axis units of a layer of a multiscale
    image, to show it in one of the image's coordinate systems.

    The affine maps the intrinsic coordinate system of the image (in which
    the layer scale and translation are defined) to the coordinate system.
    The channel axis, if any, is removed.

    Returns None if the image can't be shown in the coordinate system.
    """
    if not isinstance(multiscale, ome_zarr_models.v06.multiscales.Multiscale):
        return None
    intrinsic = multiscale.intrinsic_coordinate_system.name
    matrix: np.ndarray | None
    if coordinate_system == intrinsic:
        matrix = np.eye(multiscale.ndim + 1)
    else:
        matrix = _coordinate_system_graph(multiscale).find_matrix(
            intrinsic, coordinate_system
        )
        if matrix is None:
            return None
    affine = to_napari_affine(matrix, channel_axis)
    if affine is None:
        return None

    (cs,) = [
        cs
        for cs in multiscale.coordinateSystems
        if cs.name == coordinate_system
    ]
    axes = list(cs.axes)
    if channel_axis is not None:
        axes.pop(channel_axis)
    names = [axis.name for axis in axes]
    axis_labels = (
        None
        if any(name is None for name in names)
        else [str(n) for n in names]
    )
    axis_units = [
        "pixel" if axis.unit is None else str(axis.unit) for axis in axes
    ]
    return affine, axis_labels, axis_units


def _get_scale_translation(
    multiscale: AnyMultiscale,
) -> tuple[list[float], list[float]]:
    """
    Get the scale and translation of the highest resolution dataset,
    mapping array indices to the intrinsic physical coordinate system.
    """
    ndim = multiscale.ndim
    # datasets[0] is the highest resolution
    if isinstance(multiscale, ome_zarr_models.v06.multiscales.Multiscale):
        # 0.6 dataset transforms are a scale, identity, or sequence of
        # scale + translation, so the affine matrix is diagonal
        (transform_v06,) = multiscale.datasets[0].coordinateTransformations
        matrix = np.asarray(
            transform_v06.as_affine().affine_matrix, dtype=float
        )
        return list(np.diag(matrix[:ndim, :ndim])), list(matrix[:ndim, ndim])

    scale = [1.0] * ndim
    translation = [0.0] * ndim
    for transform in multiscale.datasets[0].coordinateTransformations:
        # Scales/translations stored in Zarr arrays are not supported,
        # so default to no scaling/translation
        if isinstance(transform, VectorScale):
            scale = list(transform.scale)
        elif isinstance(transform, VectorTranslation):
            translation = list(transform.translation)
    return scale, translation


def _get_channel_axis(multiscale: AnyMultiscale) -> int | None:
    """
    Get channel axis from Multiscale metadata.
    """
    for i, axis in enumerate(_get_axes(multiscale)):
        if axis.type == "channel":
            return i
    return None


def load_ome_zarr(
    viewer: "napari.Viewer",
    group: zarr.Group,
    *,
    visible: bool = True,
    fuse_scenes: bool = False,
) -> AddedLayers:
    """
    Load an OME-Zarr file into a napari viewer.

    The images of an OME-Zarr 0.6 scene are placed in the first coordinate
    system of the scene.

    Parameters
    ----------
    viewer : napari.Viewer
        Viewer to add data to.
    group : zarr.Group
        Open OME-Zarr group.
    visible : bool
        Set visible status of any created napari layers.
    fuse_scenes : bool
        Fuse the images of an OME-Zarr 0.6 scene into one layer per channel
        and label image, instead of separate layers per image.

    Returns
    -------
    layers :
        Dictionary mapping the layers added to their original OME-Zarr metadata models.
    """
    data = open_ome_zarr(group)

    if not isinstance(data, SUPPORTED_CLASSES):
        raise RuntimeError(
            f"Found {type(data)} - loading not currently supported."
        )

    if isinstance(data, ome_zarr_models.v06.Scene):
        scene = _read_ome_zarr_scene(viewer, group, data, visible=visible)
        coordinate_system = (
            scene.coordinate_systems[0] if scene.coordinate_systems else None
        )
        _, added_layers = scene.show(coordinate_system, fused=fuse_scenes)
        return added_layers

    return _load_ome_zarr_image(viewer, group, data, visible=visible)


def _read_ome_zarr_scene(
    viewer: "napari.Viewer",
    zarr_group: zarr.Group,
    scene: ome_zarr_models.v06.Scene,
    *,
    visible: bool = True,
) -> LoadedScene:
    """
    Read the images of an OME-Zarr 0.6 scene, without adding them to the
    viewer yet.
    """
    graph = SceneGraph(scene)
    if not graph.coordinate_systems:
        warnings.warn(
            "Scene has no coordinate systems, showing each image in its "
            "intrinsic coordinate system.",
            stacklevel=2,
        )
    specs = {
        path: _image_layer_specs(
            zarr.open_group(zarr_group.store_path / path, mode="r"),
            image,
            visible=visible,
            name=path,
        )
        for path, image in graph.images.items()
    }
    name = str(zarr_group.store_path).rstrip("/").split("/")[-1] or "scene"
    return LoadedScene(viewer, name, graph, specs)


def _load_ome_zarr_image(
    viewer: "napari.Viewer",
    zarr_group: zarr.Group,
    image: AnyImage | AnyImageLabel,
    *,
    visible: bool = True,
    layer_multiscales: dict[napari.layers.Layer, AnyMultiscale] | None = None,
) -> AddedLayers:
    """
    Load an OME-Zarr image on to the napari viewer.

    If given, ``layer_multiscales`` is updated with the multiscale image each
    added layer was created from.
    """
    added_layers: AddedLayers = {}
    for spec in _image_layer_specs(zarr_group, image, visible=visible):
        layers = spec.add_to(viewer)
        added_layers.update(layers)
        if layer_multiscales is not None:
            layer_multiscales.update(dict.fromkeys(layers, spec.multiscale))
    return added_layers


def _image_layer_specs(
    zarr_group: zarr.Group,
    image: AnyImage | AnyImageLabel,
    *,
    visible: bool = True,
    name: str | None = None,
) -> list[LayerSpec]:
    """
    Get the layers of an OME-Zarr image, including its labels.

    If name is given, it is used as the name of the image layers, and as a
    prefix for the names of the label layers.
    """
    layer_type: Literal["image", "labels"] = (
        "image" if isinstance(image, AnyImage) else "labels"
    )
    specs = [
        _multiscale_layer_spec(
            multiscale,
            zarr_group,
            image,
            layer_type=layer_type,
            visible=visible,
            name=name,
        )
        for multiscale in image.ome_attributes.multiscales
    ]

    # Check for labels
    if isinstance(image, AnyImage) and (labels := image.labels) is not None:
        for path in labels.ome_attributes.labels:
            image_label_group = zarr.open_group(
                zarr_group.store_path / "labels" / path
            )
            if image.ome_zarr_version not in IMAGE_LABEL_CLASSES:
                raise ValueError(
                    f"Can't handle OME-Zarr version {image.ome_zarr_version}"
                )
            image_label_cls = IMAGE_LABEL_CLASSES[image.ome_zarr_version]
            image_labels = image_label_cls.from_zarr(image_label_group)
            for multiscale in image_labels.ome_attributes.multiscales:
                # TODO: correctly assign color from the label metdaata
                specs.append(
                    _multiscale_layer_spec(
                        multiscale,
                        image_label_group,
                        image_labels,
                        layer_type="labels",
                        visible=visible,
                        name=None if name is None else f"{name}: {path}",
                        label=path,
                    )
                )

    return specs


def _multiscale_layer_spec(
    multiscale: AnyMultiscale,
    zarr_group: zarr.Group,
    model: AnyImage | AnyImageLabel,
    *,
    layer_type: Literal["image", "labels"],
    visible: bool = True,
    name: str | None = None,
    label: str | None = None,
) -> LayerSpec:
    """
    Get the layer of a OME-Zarr multiscales dataset.

    The layer is named after the multiscale, unless name is given.
    """
    arrays = [
        da.from_array(
            zarr.open_array(
                store=zarr_group.store_path / dataset.path,
                zarr_format=zarr_group.metadata.zarr_format,
            )
        )
        for dataset in multiscale.datasets
    ]
    axis_labels = _get_axis_names(multiscale)
    axis_units = _get_axis_units(multiscale)
    scale, translation = _get_scale_translation(multiscale)
    kwargs: dict[str, Any] = {
        "name": multiscale.name if name is None else name,
        "visible": visible,
    }
    if layer_type == "image":
        channel_axis = _get_channel_axis(multiscale)
        # If there's a channel axis, remove the associated label/units/scale
        if channel_axis is not None:
            scale.pop(channel_axis)
            translation.pop(channel_axis)
            if axis_labels is not None:
                axis_labels.pop(channel_axis)
            axis_units.pop(channel_axis)
        kwargs["channel_axis"] = channel_axis

    kwargs.update(
        axis_labels=axis_labels,
        units=axis_units,
        scale=scale,
        translate=translation,
    )
    return LayerSpec(
        data=arrays,
        layer_type=layer_type,
        kwargs=kwargs,
        model=model,
        label=label,
        multiscale=multiscale,
    )
