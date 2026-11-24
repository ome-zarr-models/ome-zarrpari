"""
Tests for loading OME-Zarr 0.6 data, using small synthetic datasets.
"""

import numpy as np
import pytest
import zarr

from ome_zarrpari._widget import OMEZarrpariWidget

SCALE = (0.5, 0.25)
TRANSLATION = (10.0, 20.0)


def _multiscale_attrs(name: str, axes: list[dict], n_levels: int) -> dict:
    """
    OME-Zarr 0.6 multiscales metadata, with each level downsampled by 2 in
    the space axes, and a translation of the whole image.
    """
    is_space = [axis["type"] == "space" for axis in axes]
    datasets = []
    for level in range(n_levels):
        factor = 2**level
        scale = [1.0] * (len(axes) - len(SCALE)) + [s * factor for s in SCALE]
        translation = [0.0] * (len(axes) - len(TRANSLATION)) + list(
            TRANSLATION
        )
        assert sum(is_space) == len(SCALE)
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {
                        "type": "sequence",
                        "input": {"path": str(level)},
                        "output": {"name": "physical"},
                        "transformations": [
                            {"type": "scale", "scale": scale},
                            {
                                "type": "translation",
                                "translation": translation,
                            },
                        ],
                    }
                ],
            }
        )
    return {
        "name": name,
        "coordinateSystems": [{"name": "physical", "axes": axes}],
        "datasets": datasets,
    }


def _write_multiscale(
    group: zarr.Group, shape: tuple[int, ...], dtype: str, n_levels: int
) -> None:
    rng = np.random.default_rng(0)
    for level in range(n_levels):
        level_shape = shape[:-2] + tuple(s // 2**level for s in shape[-2:])
        array = group.create_array(
            str(level), shape=level_shape, dtype=dtype, chunks=level_shape
        )
        array[:] = rng.integers(0, 10, size=level_shape).astype(dtype)


@pytest.fixture
def v06_image_path(tmp_path) -> str:
    """
    A 2-channel OME-Zarr 0.6 image, with a label image.
    """
    path = tmp_path / "image.ome.zarr"
    space_axes = [
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    n_levels = 2

    root = zarr.open_group(path, mode="w", zarr_format=3)
    root.attrs["ome"] = {
        "version": "0.6",
        "multiscales": [
            _multiscale_attrs(
                "my_image",
                [{"name": "c", "type": "channel"}, *space_axes],
                n_levels,
            )
        ],
    }
    _write_multiscale(root, (2, 32, 64), "uint16", n_levels)

    labels = root.create_group("labels")
    labels.attrs["ome"] = {"version": "0.6", "labels": ["cells"]}
    cells = labels.create_group("cells")
    cells.attrs["ome"] = {
        "version": "0.6",
        "image-label": {},
        "multiscales": [_multiscale_attrs("cells", space_axes, n_levels)],
    }
    _write_multiscale(cells, (32, 64), "uint8", n_levels)
    return str(path)


def test_load_v06_image(make_napari_viewer, v06_image_path) -> None:
    viewer = make_napari_viewer()
    widget = OMEZarrpariWidget(viewer)
    widget._load_ome_zarr(v06_image_path, visible=False)
    assert widget.load_pane_status_text == "Successfully loaded"

    # One layer per channel, plus the labels
    assert [layer.name for layer in viewer.layers] == [
        "my_image",
        "my_image [1]",
        "cells",
    ]
    assert [type(layer).__name__ for layer in viewer.layers] == [
        "Image",
        "Image",
        "Labels",
    ]
    assert len(widget.added_layers) == 3

    for layer in viewer.layers:
        assert layer.multiscale
        assert len(layer.data) == 2
        assert layer.data[0].shape == (32, 64)
        np.testing.assert_allclose(layer.scale, SCALE)
        np.testing.assert_allclose(layer.translate, TRANSLATION)
        assert tuple(layer.axis_labels) == ("y", "x")
        # The channel axis has no unit, which is ignored
        assert [str(unit) for unit in layer.units] == [
            "micrometer",
            "micrometer",
        ]


def test_load_v06_image_missing_unit(
    make_napari_viewer, v06_image_path
) -> None:
    """
    Axes without a unit are given the napari default unit ('pixel').
    """
    root = zarr.open_group(v06_image_path, mode="r+")
    attrs = root.attrs.asdict()
    axes = attrs["ome"]["multiscales"][0]["coordinateSystems"][0]["axes"]
    del axes[2]["unit"]
    root.attrs.put(attrs)

    viewer = make_napari_viewer()
    widget = OMEZarrpariWidget(viewer)
    widget._load_ome_zarr(v06_image_path, visible=False)
    assert widget.load_pane_status_text == "Successfully loaded"
    for name in ("my_image", "my_image [1]"):
        assert [str(unit) for unit in viewer.layers[name].units] == [
            "micrometer",
            "pixel",
        ]


STAGE_OFFSET = (100.0, 200.0)


def _add_stage_coordinate_system(
    group: zarr.Group, channel_scale: float
) -> None:
    """
    Add a 'stage' coordinate system to a 0.6 multiscale image, translated
    from the intrinsic coordinate system by STAGE_OFFSET (and scaled by
    channel_scale along a channel axis, if present).
    """
    attrs = group.attrs.asdict()
    multiscale = attrs["ome"]["multiscales"][0]
    (physical,) = multiscale["coordinateSystems"]
    axes = [
        {**axis, "name": axis["name"] + "_stage"} for axis in physical["axes"]
    ]
    multiscale["coordinateSystems"].append({"name": "stage", "axes": axes})
    n_extra = len(axes) - len(STAGE_OFFSET)
    multiscale["coordinateTransformations"] = [
        {
            "type": "sequence",
            "input": {"name": "physical"},
            "output": {"name": "stage"},
            "transformations": [
                {
                    "type": "scale",
                    "scale": [channel_scale] * n_extra + [1.0, 1.0],
                },
                {
                    "type": "translation",
                    "translation": [0.0] * n_extra + list(STAGE_OFFSET),
                },
            ],
        }
    ]
    group.attrs.put(attrs)


def test_v06_image_coordinate_systems(
    make_napari_viewer, v06_image_path
) -> None:
    root = zarr.open_group(v06_image_path, mode="r+")
    _add_stage_coordinate_system(root, channel_scale=1.0)
    _add_stage_coordinate_system(root["labels/cells"], channel_scale=1.0)

    viewer = make_napari_viewer()
    widget = OMEZarrpariWidget(viewer)
    widget._load_ome_zarr(v06_image_path, visible=False)
    dropdown = widget.image_coord_system_dropdown
    image_layers = [viewer.layers["my_image"], viewer.layers["my_image [1]"]]
    labels_layer = viewer.layers["cells"]

    # Selecting a layer lists the coordinate systems of its image
    viewer.layers.selection.active = image_layers[1]
    assert [dropdown.itemText(i) for i in range(dropdown.count())] == [
        "physical",
        "stage",
    ]
    assert dropdown.currentText() == "physical"
    assert dropdown.isEnabled()

    # Choosing a coordinate system moves all layers (channels) of the image,
    # but not the labels
    dropdown.setCurrentText("stage")
    for layer in image_layers:
        np.testing.assert_allclose(
            layer.data_to_world((0, 0)),
            np.add(TRANSLATION, STAGE_OFFSET),
        )
        assert tuple(layer.axis_labels) == ("y_stage", "x_stage")
    np.testing.assert_allclose(labels_layer.data_to_world((0, 0)), TRANSLATION)

    # The chosen coordinate system is remembered per image
    viewer.layers.selection.active = labels_layer
    assert dropdown.currentText() == "physical"
    viewer.layers.selection.active = image_layers[0]
    assert dropdown.currentText() == "stage"

    # And can be switched back
    dropdown.setCurrentText("physical")
    for layer in image_layers:
        np.testing.assert_allclose(layer.data_to_world((0, 0)), TRANSLATION)
        assert tuple(layer.axis_labels) == ("y", "x")


def test_v06_image_coordinate_system_changing_channels(
    make_napari_viewer, v06_image_path
) -> None:
    """
    Coordinate systems that change the channel axis can't be shown in napari.
    """
    root = zarr.open_group(v06_image_path, mode="r+")
    _add_stage_coordinate_system(root, channel_scale=2.0)
    _add_stage_coordinate_system(root["labels/cells"], channel_scale=2.0)

    viewer = make_napari_viewer()
    widget = OMEZarrpariWidget(viewer)
    widget._load_ome_zarr(v06_image_path, visible=False)
    dropdown = widget.image_coord_system_dropdown

    viewer.layers.selection.active = viewer.layers["my_image"]
    dropdown.setCurrentText("stage")
    assert widget.load_pane_status_text.startswith("Can't show 'my_image'")
    assert dropdown.currentText() == "physical"
    np.testing.assert_allclose(
        viewer.layers["my_image"].data_to_world((0, 0)), TRANSLATION
    )

    # The labels have no channel axis, so can be shown
    viewer.layers.selection.active = viewer.layers["cells"]
    dropdown.setCurrentText("stage")
    np.testing.assert_allclose(
        viewer.layers["cells"].data_to_world((0, 0)),
        np.add(TRANSLATION, STAGE_OFFSET),
    )
