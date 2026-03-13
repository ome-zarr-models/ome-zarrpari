from typing import TYPE_CHECKING, Literal

import napari.layers
import ome_zarr_models.v04
import ome_zarr_models.v04.multiscales
import ome_zarr_models.v05.multiscales
import zarr
from napari.settings import get_settings
from ome_zarr_models import open_ome_zarr
from ome_zarr_models.common.coordinate_transformations import VectorScale
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import napari

from qtpy.QtWidgets import QLabel

AnyImage = ome_zarr_models.v04.Image | ome_zarr_models.v05.Image
AnyImageLabel = ome_zarr_models.v04.ImageLabel | ome_zarr_models.v05.ImageLabel

SUPPORTED_CLASSES = (AnyImage, AnyImageLabel)

AnyMultiscale = (
    ome_zarr_models.v04.multiscales.Multiscale
    | ome_zarr_models.v05.multiscales.Multiscale
)


class OMEZarrpariWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        # Keep track of layers added by this widget
        self.added_layers = {}

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
            self.viewer, group, data, visible=visible
        )
        self.added_layers.update(added_layers)
        self.status_text.setText("Successfully loaded")


def _get_axis_names(multiscale: AnyMultiscale) -> list[str] | None:
    """
    Get axis labels from Multiscale metadata.
    """
    axis_labels_raw = [axis.name for axis in multiscale.axes]
    if any(label is None for label in axis_labels_raw):
        print(
            f"Warning: At least one axis label is None for multiscale '{multiscale.name}', "
            "not setting any axis labels."
        )
        return None
    else:
        return [str(label) for label in axis_labels_raw]


def _get_axis_units(multiscale: AnyMultiscale) -> list[str] | None:
    """
    Get axis units from Multiscale metadata.

    # TODO: convert strings to pint units if they make sense as physical units
    """
    axis_units_raw = [axis.unit for axis in multiscale.axes]
    if any(unit is None for unit in axis_units_raw):
        print(
            f"Warning: At least one unit is None for multiscale '{multiscale.name}', "
            "not setting any axis units."
        )
        return None
    else:
        return [str(unit) for unit in axis_units_raw]


def _get_scale(multiscale: AnyMultiscale) -> list[float]:
    # datasets[0] is the highest resolution
    scale_transform = multiscale.datasets[0].coordinateTransformations[0]
    if isinstance(scale_transform, VectorScale):
        return scale_transform.scale
    else:
        # Scale is stored in a Zarr array, default to no scaling
        return [1] * multiscale.ndim


def _get_channel_axis(multiscale: AnyMultiscale) -> int | None:
    """
    Get channel axis from Multiscale metadata.
    """
    for i, axis in enumerate(multiscale.axes):
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
) -> dict[
    napari.layers.Image | napari.layers.Labels, AnyImage | AnyImageLabel
]:
    """
    Load an OME-Zarr image on to the napari viewer.
    """
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
        if isinstance(layer, list):
            for layer_ in layer:
                added_layers[layer_] = image
        else:
            added_layers[layer] = image

    # Check for labels
    if isinstance(image, AnyImage) and (labels := image.labels) is not None:
        for path in labels.ome_attributes.labels:
            image_label_group = zarr.open_group(
                zarr_group.store_path / "labels" / path
            )
            image_labels = ome_zarr_models.v04.ImageLabel.from_zarr(
                image_label_group
            )
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
        zarr.open_array(
            store=zarr_group.store_path / dataset.path,
            zarr_format=zarr_group.metadata.zarr_format,
        )
        for dataset in multiscale.datasets
    ]
    axis_labels = _get_axis_names(multiscale)
    axis_units = _get_axis_units(multiscale)
    scale = _get_scale(multiscale)
    if layer_type == "image":
        channel_axis = _get_channel_axis(multiscale)
        # If there's a channel axis, remove the associated label/units/scale
        if channel_axis is not None:
            scale.pop(channel_axis)
            if axis_labels is not None:
                axis_labels.pop(channel_axis)
            if axis_units is not None:
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
        )
