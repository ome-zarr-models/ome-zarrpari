from typing import TYPE_CHECKING, Literal

import ome_zarr_models.v04
import ome_zarr_models.v04.multiscales
import ome_zarr_models.v05.multiscales
import zarr
from ome_zarr_models import open_ome_zarr
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
            group = zarr.open_group(path)
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
        multiscale: (
            ome_zarr_models.v04.multiscales.Multiscale
            | ome_zarr_models.v05.multiscales.Multiscale
        ),
        zarr_group: zarr.Group,
        *,
        layer_type: Literal["image", "labels"],
        visible: bool = True,
    ) -> None:
        arrays = []
        for dataset in multiscale.datasets:
            array_store = zarr_group.store_path / dataset.path
            arrays.append(
                zarr.open_array(
                    array_store,
                    zarr_format=(
                        2
                        if isinstance(
                            multiscale,
                            ome_zarr_models.v04.multiscales.Multiscale,
                        )
                        else 3
                    ),
                )
            )

        # TODO: pass axis labels down
        if layer_type == "image":
            self.viewer.add_image(
                arrays, name=multiscale.name, multiscale=True, visible=visible
            )
        elif layer_type == "labels":
            self.viewer.add_labels(
                arrays, name=multiscale.name, multiscale=True, visible=visible
            )
