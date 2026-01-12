from typing import TYPE_CHECKING

import ome_zarr_models.v04
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

SUPPORTED_CLASSES = (ome_zarr_models.v05.Image,)


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
        for multiscale in image.ome_attributes.multiscales:
            arrays = []
            for dataset in multiscale.datasets:
                array_store = zarr_group.store_path / dataset.path
                arrays.append(
                    zarr.open_array(
                        array_store,
                        zarr_format=2 if image.ome_zarr_version == 0.4 else 3,
                    )
                )

            self.viewer.add_image(
                arrays, name=multiscale.name, multiscale=True, visible=visible
            )
        self.status_text.setText("Successfully loaded")
