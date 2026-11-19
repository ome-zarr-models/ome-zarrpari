"""
Tests for applying the scale and translation of OME-Zarr 0.4 and 0.5 images.
"""

import numpy as np
import pytest
import zarr

from ome_zarrpari._widget import OMEZarrpariWidget

SCALE = [0.5, 0.25]
TRANSLATION = [10.0, 20.0]


def _multiscales(version: str) -> list[dict]:
    datasets = [
        {
            "path": str(level),
            "coordinateTransformations": [
                {"type": "scale", "scale": [s * 2**level for s in SCALE]},
                {"type": "translation", "translation": TRANSLATION},
            ],
        }
        for level in range(2)
    ]
    multiscale = {
        "name": "translated",
        "axes": [
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "datasets": datasets,
    }
    if version == "0.4":
        multiscale["version"] = "0.4"
    return [multiscale]


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_scale_translation(make_napari_viewer, tmp_path, version) -> None:
    path = tmp_path / "image.ome.zarr"
    if version == "0.4":
        root = zarr.open_group(path, mode="w", zarr_format=2)
        root.attrs["multiscales"] = _multiscales(version)
    else:
        root = zarr.open_group(path, mode="w", zarr_format=3)
        root.attrs["ome"] = {
            "version": version,
            "multiscales": _multiscales(version),
        }
    for level in range(2):
        shape = (32 // 2**level, 64 // 2**level)
        # OME-Zarr 0.5 requires dimension names, Zarr v2 doesn't support them
        dimension_names = None if version == "0.4" else ["y", "x"]
        root.create_array(
            str(level),
            shape=shape,
            dtype="uint8",
            dimension_names=dimension_names,
        )[:] = 1

    viewer = make_napari_viewer()
    widget = OMEZarrpariWidget(viewer)
    widget._load_ome_zarr(str(path), visible=False)
    assert widget.load_pane_status_text == "Successfully loaded"

    (layer,) = viewer.layers
    np.testing.assert_allclose(layer.scale, SCALE)
    np.testing.assert_allclose(layer.translate, TRANSLATION)
