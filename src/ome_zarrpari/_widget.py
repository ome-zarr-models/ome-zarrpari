from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

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
from ome_zarr_models import open_ome_zarr
from ome_zarr_models.common.coordinate_transformations import (
    VectorScale,
    VectorTranslation,
)
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

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

SUPPORTED_CLASSES = (AnyImage, AnyImageLabel)

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


class OMEZarrpariWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        # Keep track of layers added by this widget
        self.added_layers = {}
        # Multiscale image each added layer was created from
        self._layer_multiscales: dict[napari.layers.Layer, AnyMultiscale] = {}
        # Coordinate system each added layer is shown in, if not intrinsic
        self._layer_coordinate_systems: dict[napari.layers.Layer, str] = {}

        # Create text box and button
        self.text_box = QLineEdit()
        self.text_box.setPlaceholderText("Enter OME-Zarr path or URL...")

        # Browse button
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self._on_browse)

        btn = QPushButton("Load OME-Zarr")
        btn.clicked.connect(self._on_load)

        # Create label for status messages
        self.status_text = QLabel("")
        self.status_text.setWordWrap(True)

        # Add widgets to vertical layout
        layout = QVBoxLayout()

        text_row = QHBoxLayout()
        text_row.addWidget(self.text_box)
        text_row.addWidget(self.browse_btn)
        layout.addLayout(text_row)

        layout.addWidget(btn)
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
        path = self.text_box.text()
        self._load_ome_zarr(path)

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
    viewer: "napari.Viewer", group: zarr.Group, *, visible: bool = True
) -> dict[
    napari.layers.Image | napari.layers.Labels, AnyImage | AnyImageLabel
]:
    """
    Load an OME-Zarr file into a napari viewer.

    Parameters
    ----------
    viewer : napari.Viewer
        Viewer to add data to.
    group : zarr.Group
        Open OME-Zarr group.
    visible : bool
        Set visible status of any created napari layers.

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

    return _load_ome_zarr_image(viewer, group, data, visible=visible)


def _load_ome_zarr_image(
    viewer: "napari.Viewer",
    zarr_group: zarr.Group,
    image: AnyImage | AnyImageLabel,
    *,
    visible: bool = True,
    layer_multiscales: dict[napari.layers.Layer, AnyMultiscale] | None = None,
) -> dict[
    napari.layers.Image | napari.layers.Labels, AnyImage | AnyImageLabel
]:
    """
    Load an OME-Zarr image on to the napari viewer.

    If given, ``layer_multiscales`` is updated with the multiscale image each
    added layer was created from.
    """
    if layer_multiscales is None:
        layer_multiscales = {}
    layer_type: Literal["image", "labels"] = (
        "image" if isinstance(image, AnyImage) else "labels"
    )
    added_layers = {}
    # Add all the images
    for multiscale in image.ome_attributes.multiscales:
        layer = _add_multiscale_layer(
            viewer,
            multiscale,
            zarr_group,
            layer_type=layer_type,
            visible=visible,
        )
        for layer_ in layer if isinstance(layer, list) else [layer]:
            added_layers[layer_] = image
            layer_multiscales[layer_] = multiscale

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
                layer = _add_multiscale_layer(
                    viewer,
                    multiscale,
                    image_label_group,
                    layer_type="labels",
                    visible=visible,
                )
                added_layers[layer] = image_labels
                layer_multiscales[layer] = multiscale

    return added_layers


def _add_multiscale_layer(
    viewer: "napari.Viewer",
    multiscale: AnyMultiscale,
    zarr_group: zarr.Group,
    *,
    layer_type: Literal["image", "labels"],
    visible: bool = True,
) -> napari.layers.Image | list[napari.layers.Image] | napari.layers.Labels:
    """
    Add a OME-Zarr multiscales dataset to the napari viewer.
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
    if layer_type == "image":
        channel_axis = _get_channel_axis(multiscale)
        # If there's a channel axis, remove the associated label/units/scale
        if channel_axis is not None:
            scale.pop(channel_axis)
            translation.pop(channel_axis)
            if axis_labels is not None:
                axis_labels.pop(channel_axis)
            axis_units.pop(channel_axis)

        return viewer.add_image(
            arrays,
            name=multiscale.name,
            multiscale=True,
            visible=visible,
            axis_labels=axis_labels,
            channel_axis=channel_axis,
            units=axis_units,
            scale=scale,
            translate=translation,
        )
    else:
        return viewer.add_labels(
            arrays,
            name=multiscale.name,
            multiscale=True,
            visible=visible,
            axis_labels=axis_labels,
            units=axis_units,
            scale=scale,
            translate=translation,
        )
