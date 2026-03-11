from typing import TYPE_CHECKING, Literal

import ome_zarr_models.v04
import ome_zarr_models.v04.multiscales
import ome_zarr_models.v05.multiscales
import zarr
from ome_zarr_models import open_ome_zarr
from ome_zarr_models.common.coordinate_transformations import VectorScale
from qtpy.QtWidgets import (
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import napari

from qtpy.QtWidgets import QLabel

SUPPORTED_CLASSES = (
    ome_zarr_models.v04.Image,
    ome_zarr_models.v05.Image,
)

AnyMultiscale = (
    ome_zarr_models.v04.multiscales.Multiscale
    | ome_zarr_models.v05.multiscales.Multiscale
)


class OMEZarrpariWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        # Create text box and button
        self.text_box = QLineEdit()
        self.text_box.setPlaceholderText("Enter OME-Zarr path or URL...")

        btn = QPushButton("Load OME-Zarr")
        btn.clicked.connect(self._on_load)

        # Create label for status messages
        self.status_text = QLabel("")
        self.status_text.setWordWrap(True)

        # Add widgets to vertical layout
        layout = QVBoxLayout()
        layout.addWidget(self.text_box)
        layout.addWidget(btn)
        layout.addWidget(self.status_text)
        layout.addStretch()  # Push everything to the top
        self.setLayout(layout)

    @property
    def load_pane_status_text(self) -> str:
        return self.status_text.text()

    def _on_load(self) -> None:
        path = self.text_box.text()
        self._load_ome_zarr(path)

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

        self._load_ome_zarr_image(group, data, visible=visible)
        self.status_text.setText("Successfully loaded")

    def _load_ome_zarr_image(
        self,
        zarr_group: zarr.Group,
        image: ome_zarr_models.v04.Image | ome_zarr_models.v05.Image,
        *,
        visible: bool = True,
    ) -> None:
        """
        Load an OME-Zarr image on to the napari viewer.
        """
        # Add all the images

        for multiscale in image.ome_attributes.multiscales:
            self._add_multiscale_layer(
                multiscale, zarr_group, layer_type="image", visible=visible
            )

        # Check for labels
        if (labels := image.labels) is not None:
            for path in labels.ome_attributes.labels:
                image_label_group = zarr.open_group(
                    zarr_group.store_path / "labels" / path
                )
                image_labels = ome_zarr_models.v04.ImageLabel.from_zarr(
                    image_label_group
                )
                for multiscale in image_labels.ome_attributes.multiscales:
                    # TODO: correctly assign color from the label metdaata
                    self._add_multiscale_layer(
                        multiscale,
                        image_label_group,
                        layer_type="labels",
                        visible=visible,
                    )

    def _add_multiscale_layer(
        self,
        multiscale: AnyMultiscale,
        zarr_group: zarr.Group,
        *,
        layer_type: Literal["image", "labels"],
        visible: bool = True,
    ) -> None:
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

            self.viewer.add_image(
                arrays,
                name=multiscale.name,
                multiscale=True,
                visible=visible,
                axis_labels=axis_labels,
                channel_axis=channel_axis,
                units=axis_units,
                scale=scale,
            )
        elif layer_type == "labels":
            self.viewer.add_labels(
                arrays,
                name=multiscale.name,
                multiscale=True,
                visible=visible,
                axis_labels=axis_labels,
                units=axis_units,
                scale=scale,
            )


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
