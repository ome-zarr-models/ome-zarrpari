"""
Tests for loading OME-Zarr 0.6 data, using small synthetic datasets.
"""

import napari.layers
import numpy as np
import ome_zarr_models.v06
import pytest
import zarr
from ome_zarr_models import open_ome_zarr
from ome_zarr_models.v06.coordinate_transforms import (
    CoordinateSystemIdentifier,
    Translation,
)

from ome_zarrpari._scene import SceneGraph
from ome_zarrpari._transforms import to_napari_affine
from ome_zarrpari._widget import (
    OMEZarrpariWidget,
    _channel_display,
    load_ome_zarr,
)

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


SPACE_AXES = [
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "x", "type": "space", "unit": "micrometer"},
]
CHANNEL_AXES = [{"name": "c", "type": "channel"}, *SPACE_AXES]
N_LEVELS = 2


def _write_v06_image(
    group: zarr.Group,
    name: str,
    *,
    with_labels: bool = True,
    dtype: str = "uint16",
) -> None:
    """
    Write a 2-channel OME-Zarr 0.6 image to a group, optionally with
    a label image.
    """
    group.attrs["ome"] = {
        "version": "0.6",
        "multiscales": [_multiscale_attrs(name, CHANNEL_AXES, N_LEVELS)],
    }
    _write_multiscale(group, (2, 32, 64), dtype, N_LEVELS)
    if not with_labels:
        return

    labels = group.create_group("labels")
    labels.attrs["ome"] = {"version": "0.6", "labels": ["cells"]}
    cells = labels.create_group("cells")
    cells.attrs["ome"] = {
        "version": "0.6",
        "image-label": {},
        "multiscales": [_multiscale_attrs("cells", SPACE_AXES, N_LEVELS)],
    }
    _write_multiscale(cells, (32, 64), "uint8", N_LEVELS)


@pytest.fixture
def v06_image_path(tmp_path) -> str:
    """
    A 2-channel OME-Zarr 0.6 image, with a label image.
    """
    path = tmp_path / "image.ome.zarr"
    root = zarr.open_group(path, mode="w", zarr_format=3)
    _write_v06_image(root, "my_image")
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


# Scenes

# Rotation by 90 degrees in the yx plane, as a (c, y, x) affine
ROTATION = [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0]]


def _translation(
    translation: list[float], *, source: dict, target: dict
) -> dict:
    return {
        "type": "translation",
        "translation": translation,
        "input": source,
        "output": target,
    }


def _write_v06_scene(
    path,
    transforms: list[dict],
    tiles: tuple[str, ...],
    coordinate_systems: tuple[str, ...] = ("world", "rotated"),
    dtypes: dict[str, str] | None = None,
) -> str:
    """
    Write an OME-Zarr 0.6 scene with the given image tiles, and (c, y, x)
    scene coordinate systems.
    """
    root = zarr.open_group(path, mode="w", zarr_format=3)
    root.attrs["ome"] = {
        "version": "0.6",
        "scene": {
            "coordinateSystems": [
                {"name": name, "axes": CHANNEL_AXES}
                for name in coordinate_systems
            ],
            "coordinateTransformations": transforms,
        },
    }
    for tile in tiles:
        _write_v06_image(
            root.create_group(tile),
            "my_image",
            with_labels=tile == "tile0",
            dtype=(dtypes or {}).get(tile, "uint16"),
        )
    return str(path)


@pytest.fixture
def v06_scene_path(tmp_path) -> str:
    """
    A scene with three tiles:

    - tile0 (with labels), translated by (5, 0) into 'world'
    - tile1, translated by (0, 100) into 'world'
    - tile2, with a transform from 'world' to tile2, translated by (50, 50),
      so it is placed by the inverse of that transform

    and 'world' rotated by 90 degrees into 'rotated'.
    """
    world = {"name": "world"}
    transforms = [
        _translation(
            [0, 5, 0],
            source={"path": "tile0", "name": "physical"},
            target=world,
        ),
        _translation(
            [0, 0, 100],
            source={"path": "tile1", "name": "physical"},
            target=world,
        ),
        _translation(
            [0, 50, 50],
            source=world,
            target={"path": "tile2", "name": "physical"},
        ),
        {
            "type": "affine",
            "affine": ROTATION,
            "input": world,
            "output": {"name": "rotated"},
        },
    ]
    return _write_v06_scene(
        tmp_path / "scene.ome.zarr", transforms, ("tile0", "tile1", "tile2")
    )


def _affine_translation(layer) -> tuple[float, ...]:
    return tuple(layer.affine.translate)


def test_load_v06_scene(make_napari_viewer, v06_scene_path) -> None:
    viewer = make_napari_viewer()
    widget = OMEZarrpariWidget(viewer)
    widget.fuse_checkbox.setChecked(False)
    widget._load_ome_zarr(v06_scene_path, visible=False)
    assert widget.load_pane_status_text == "Successfully loaded"

    # A layer per channel of each tile, plus the labels of tile0
    assert [layer.name for layer in viewer.layers] == [
        "tile0",
        "tile0 [1]",
        "tile0: cells",
        "tile1",
        "tile1 [1]",
        "tile2",
        "tile2 [1]",
    ]
    assert len(widget.added_layers) == 7

    # Intrinsic scale and translation are kept
    for layer in viewer.layers:
        np.testing.assert_allclose(layer.scale, SCALE)
        np.testing.assert_allclose(layer.translate, TRANSLATION)

    # Placed in the first scene coordinate system, 'world'
    assert widget.coord_system_dropdown.isEnabled()
    assert [
        widget.coord_system_dropdown.itemText(i)
        for i in range(widget.coord_system_dropdown.count())
    ] == ["world", "rotated"]
    assert widget.coord_system_dropdown.currentText() == "world"
    expected = {
        "tile0": (5, 0),
        "tile0 [1]": (5, 0),
        "tile0: cells": (5, 0),
        "tile1": (0, 100),
        "tile1 [1]": (0, 100),
        "tile2": (-50, -50),
        "tile2 [1]": (-50, -50),
    }
    for layer in viewer.layers:
        np.testing.assert_allclose(
            _affine_translation(layer), expected[layer.name]
        )
        np.testing.assert_allclose(layer.affine.linear_matrix, np.eye(2))

    # Switch to the rotated coordinate system
    widget.coord_system_dropdown.setCurrentText("rotated")
    rotation = np.array([[0, -1], [1, 0]])
    for layer in viewer.layers:
        np.testing.assert_allclose(layer.affine.linear_matrix, rotation)
        np.testing.assert_allclose(
            _affine_translation(layer), rotation @ expected[layer.name]
        )


def test_load_v06_scene_non_affine_transform(
    make_napari_viewer, tmp_path
) -> None:
    """
    An image whose transform to the scene coordinate system can't be
    converted to an affine is shown in its intrinsic coordinate system,
    with a warning.
    """
    transforms = [
        _translation(
            [0, 0, 100],
            source={"path": "tile0", "name": "physical"},
            target={"name": "world"},
        ),
        {
            "type": "byDimension",
            "transformations": [
                {
                    "transformation": {
                        "type": "translation",
                        "translation": [100],
                    },
                    "inputAxes": [2],
                    "outputAxes": [2],
                }
            ],
            "input": {"path": "tile1", "name": "physical"},
            "output": {"name": "world"},
        },
    ]
    path = _write_v06_scene(
        tmp_path / "scene.ome.zarr", transforms, ("tile0", "tile1")
    )
    viewer = make_napari_viewer()
    widget = OMEZarrpariWidget(viewer)
    widget.fuse_checkbox.setChecked(False)
    with pytest.warns(UserWarning) as record:
        widget._load_ome_zarr(path, visible=False)
    messages = [str(warning.message) for warning in record]
    assert len(messages) == 2
    assert messages[0].startswith(
        "Ignoring 'byDimension' transform from ('tile1', 'physical') to "
        "('', 'world'): it can't be converted to an affine transform"
    )
    assert messages[1].startswith(
        "Could not place layers ['tile1', 'tile1 [1]'] in scene coordinate "
        "system 'world'"
    )

    assert widget.load_pane_status_text == "Successfully loaded"
    np.testing.assert_allclose(
        _affine_translation(viewer.layers["tile0"]), (0, 100)
    )
    np.testing.assert_allclose(
        viewer.layers["tile1"].affine.affine_matrix, np.eye(3)
    )

    # There are no transforms to the 'rotated' coordinate system
    with pytest.warns(UserWarning, match=r"Could not place layers .*'tile1'"):
        widget.coord_system_dropdown.setCurrentText("rotated")


def test_load_v06_scene_layer_removed(
    make_napari_viewer, v06_scene_path
) -> None:
    """
    Changing the coordinate system skips layers removed from the viewer.
    """
    viewer = make_napari_viewer()
    widget = OMEZarrpariWidget(viewer)
    widget.fuse_checkbox.setChecked(False)
    widget._load_ome_zarr(v06_scene_path, visible=False)
    viewer.layers.remove("tile1")

    widget.coord_system_dropdown.setCurrentText("rotated")
    assert widget.scene is not None
    assert "tile1" not in [layer.name for layer in widget.scene.layers]
    assert "tile1" not in [layer.name for layer in widget.added_layers]
    np.testing.assert_allclose(
        viewer.layers["tile1 [1]"].affine.linear_matrix, [[0, -1], [1, 0]]
    )


@pytest.mark.parametrize("coordinate_systems", [("world",), ()])
def test_load_v06_scene_coordinate_systems(
    make_napari_viewer, tmp_path, coordinate_systems
) -> None:
    """
    The coordinate system dropdown is only enabled if there is a choice.
    Without scene coordinate systems, images are shown in their intrinsic
    coordinate system.
    """
    transforms = [
        _translation(
            [0, 0, 100],
            source={"path": "tile0", "name": "physical"},
            target={"name": "world"},
        )
    ]
    path = _write_v06_scene(
        tmp_path / "scene.ome.zarr",
        transforms,
        ("tile0",),
        coordinate_systems=coordinate_systems,
    )
    viewer = make_napari_viewer()
    widget = OMEZarrpariWidget(viewer)
    widget.fuse_checkbox.setChecked(False)
    if coordinate_systems:
        widget._load_ome_zarr(path, visible=False)
        expected_translation = (0, 100)
    else:
        with pytest.warns(
            UserWarning, match="Scene has no coordinate systems"
        ):
            widget._load_ome_zarr(path, visible=False)
        expected_translation = (0, 0)

    assert not widget.coord_system_dropdown.isEnabled()
    assert widget.coord_system_dropdown.count() == len(coordinate_systems)
    np.testing.assert_allclose(
        _affine_translation(viewer.layers["tile0"]), expected_translation
    )


def test_load_ome_zarr_scene(make_napari_viewer, v06_scene_path) -> None:
    """
    The public load_ome_zarr function places scene images in the first scene
    coordinate system.
    """
    viewer = make_napari_viewer()
    layers = load_ome_zarr(
        viewer, zarr.open_group(v06_scene_path, mode="r"), visible=False
    )
    assert len(layers) == 7
    np.testing.assert_allclose(
        _affine_translation(viewer.layers["tile1"]), (0, 100)
    )


def _scene_graph(tmp_path, transforms: list[dict]) -> SceneGraph:
    path = _write_v06_scene(
        tmp_path / "scene.ome.zarr", transforms, ("tile0",)
    )
    scene = open_ome_zarr(path)
    assert isinstance(scene, ome_zarr_models.v06.Scene)
    return SceneGraph(scene)


def test_scene_graph_path_only_identifier(tmp_path) -> None:
    """
    An identifier with only an image path refers to the image's intrinsic
    coordinate system.
    """
    graph = _scene_graph(
        tmp_path,
        [
            _translation(
                [0, 0, 100], source={"path": "tile0"}, target={"name": "world"}
            )
        ],
    )
    matrix = graph.image_to_scene("tile0", "world")
    assert matrix is not None
    np.testing.assert_allclose(matrix[:-1, -1], (0, 0, 100))


def test_scene_graph_singular_affine(tmp_path) -> None:
    """
    A singular affine transform can only be followed forwards.
    """
    graph = _scene_graph(
        tmp_path,
        [
            {
                "type": "affine",
                "affine": [[0, 0, 0, 0]] * 3,
                "input": {"path": "tile0", "name": "physical"},
                "output": {"name": "world"},
            }
        ],
    )
    assert graph.image_to_scene("tile0", "world") is not None
    assert ("", "world") not in graph._edges


def test_scene_graph_dimension_mismatch(tmp_path) -> None:
    """
    Paths through transforms with mismatching dimensions are not followed.
    """
    graph = _scene_graph(
        tmp_path,
        [
            # (c, y, x) -> (y, x)
            {
                "type": "affine",
                "affine": [[0, 1, 0, 0], [0, 0, 1, 0]],
                "input": {"path": "tile0", "name": "physical"},
                "output": {"name": "plane"},
            },
            # (c, y, x) -> (c, y, x)
            _translation(
                [0, 0, 100],
                source={"name": "plane"},
                target={"name": "world"},
            ),
        ],
    )
    assert graph.image_to_scene("tile0", "world") is None


def test_scene_graph_unresolved(tmp_path) -> None:
    """
    Transforms with coordinate systems that can't be resolved are ignored,
    with a warning.
    """
    graph = _scene_graph(
        tmp_path,
        [
            _translation(
                [0, 0, 100],
                source={"path": "tile0", "name": "physical"},
                target={"name": "world"},
            )
        ],
    )
    assert graph._resolve(None) is None
    assert graph._resolve(CoordinateSystemIdentifier(path="other")) is None
    with pytest.warns(UserWarning, match="Ignoring 'translation' transform"):
        graph.add_transform(Translation(translation=[0, 0, 1]), None, None)


def test_to_napari_affine() -> None:
    matrix = np.eye(4)
    matrix[1:3, 3] = (10, 20)
    np.testing.assert_allclose(
        to_napari_affine(matrix, channel_axis=0),
        [[1, 0, 10], [0, 1, 20], [0, 0, 1]],
    )
    np.testing.assert_allclose(to_napari_affine(matrix, None), matrix)
    # Not square
    assert to_napari_affine(matrix[1:], None) is None
    # Channel axis is scaled
    matrix[0, 0] = 2
    assert to_napari_affine(matrix, channel_axis=0) is None


def test_load_v06_scene_image_transform(make_napari_viewer, tmp_path) -> None:
    """
    Transforms between coordinate systems of a scene image are followed.
    """
    transforms = [
        _translation(
            [0, 0, 100],
            source={"path": "tile0", "name": "aligned"},
            target={"name": "world"},
        )
    ]
    path = _write_v06_scene(
        tmp_path / "scene.ome.zarr", transforms, ("tile0",)
    )
    # Add an 'aligned' coordinate system to tile0, translated by (1, 2)
    tile = zarr.open_group(path, mode="r+")["tile0"]
    attrs = tile.attrs.asdict()
    multiscale = attrs["ome"]["multiscales"][0]
    multiscale["coordinateSystems"].append(
        {"name": "aligned", "axes": CHANNEL_AXES}
    )
    multiscale["coordinateTransformations"] = [
        _translation(
            [0, 1, 2],
            source={"name": "physical"},
            target={"name": "aligned"},
        )
    ]
    tile.attrs.put(attrs)

    viewer = make_napari_viewer()
    load_ome_zarr(viewer, zarr.open_group(path, mode="r"), visible=False)
    np.testing.assert_allclose(
        _affine_translation(viewer.layers["tile0"]), (1, 102)
    )


def test_load_ome_zarr_image(make_napari_viewer, v06_image_path) -> None:
    viewer = make_napari_viewer()
    layers = load_ome_zarr(
        viewer, zarr.open_group(v06_image_path, mode="r"), visible=False
    )
    assert [layer.name for layer in layers] == [
        "my_image",
        "my_image [1]",
        "cells",
    ]


# Fused scenes


def _tile_data(scene_path: str, tile: str, channel: int | None) -> np.ndarray:
    root = zarr.open_group(scene_path, mode="r")
    if channel is None:
        return root[tile]["labels"]["cells"]["0"][:]
    return root[tile]["0"][channel]


def test_load_v06_scene_fused(make_napari_viewer, v06_scene_path) -> None:
    viewer = make_napari_viewer()
    widget = OMEZarrpariWidget(viewer)
    # Fused by default
    assert widget.fuse_checkbox.isChecked()
    widget._load_ome_zarr(v06_scene_path, visible=False)
    assert widget.load_pane_status_text == "Successfully loaded"

    # A layer per channel, and per label image
    assert [layer.name for layer in viewer.layers] == [
        "scene.ome.zarr [0]",
        "scene.ome.zarr [1]",
        "scene.ome.zarr: cells",
    ]
    assert set(widget.added_layers) == set(viewer.layers)
    image_0, image_1, labels = viewer.layers
    assert [layer.colormap.name for layer in (image_0, image_1)] == [
        "magenta",
        "green",
    ]
    assert isinstance(labels, napari.layers.Labels)

    # The tiles are at (5, 0), (0, 100) and (-50, -50) in world, plus their
    # intrinsic translation; so in fused pixels (with the intrinsic pixel
    # size) at (110, 200), (100, 600) and (0, 0)
    for layer in viewer.layers:
        assert len(layer.data) == N_LEVELS
        np.testing.assert_allclose(layer.affine.affine_matrix, np.eye(3))
        np.testing.assert_allclose(layer.scale, SCALE)
    np.testing.assert_allclose(image_0.translate, (-40, -30))
    for channel, layer in enumerate((image_0, image_1)):
        fused = layer.data[0].compute()
        assert fused.shape == (142, 664)
        for tile, (y, x) in {
            "tile0": (110, 200),
            "tile1": (100, 600),
            "tile2": (0, 0),
        }.items():
            np.testing.assert_array_equal(
                fused[y : y + 32, x : x + 64],
                _tile_data(v06_scene_path, tile, channel),
            )
    # Only tile0 has labels
    np.testing.assert_allclose(labels.translate, (15, 20))
    np.testing.assert_array_equal(
        labels.data[0].compute(), _tile_data(v06_scene_path, "tile0", None)
    )

    # Switching the coordinate system replaces the fused layers
    widget.coord_system_dropdown.setCurrentText("rotated")
    assert len(viewer.layers) == 3
    assert set(widget.added_layers) == set(viewer.layers)
    # Rotated by 90 degrees: the fused pixel size is swapped too
    assert viewer.layers[0].data[0].shape == (664, 142)
    np.testing.assert_allclose(viewer.layers[0].scale, SCALE[::-1])

    # The checkbox only applies when loading
    widget.fuse_checkbox.setChecked(False)
    assert len(viewer.layers) == 3

    # Loading unfused shows separate layers, in the selected coordinate
    # system; switching the coordinate system keeps them separate
    viewer.layers.clear()
    widget._load_ome_zarr(v06_scene_path, visible=False)
    assert len(viewer.layers) == 7
    widget.coord_system_dropdown.setCurrentText("rotated")
    assert len(viewer.layers) == 7
    assert set(viewer.layers) <= set(widget.added_layers)
    np.testing.assert_allclose(
        viewer.layers["tile1"].affine.linear_matrix, [[0, -1], [1, 0]]
    )


def test_load_v06_scene_fused_non_affine_transform(
    make_napari_viewer, tmp_path
) -> None:
    """
    Images that can't be placed are shown as separate layers when fusing.
    """
    transforms = [
        _translation(
            [0, 0, 100],
            source={"path": "tile0", "name": "physical"},
            target={"name": "world"},
        ),
        {
            "type": "byDimension",
            "transformations": [
                {
                    "transformation": {
                        "type": "translation",
                        "translation": [100],
                    },
                    "inputAxes": [2],
                    "outputAxes": [2],
                }
            ],
            "input": {"path": "tile1", "name": "physical"},
            "output": {"name": "world"},
        },
    ]
    path = _write_v06_scene(
        tmp_path / "scene.ome.zarr", transforms, ("tile0", "tile1")
    )
    viewer = make_napari_viewer()
    with pytest.warns(UserWarning) as record:
        load_ome_zarr(
            viewer,
            zarr.open_group(path, mode="r"),
            visible=False,
            fuse_scenes=True,
        )
    assert str(record[-1].message).startswith(
        "Could not place layers ['tile1', 'tile1 [1]']"
    )
    assert [layer.name for layer in viewer.layers] == [
        "scene.ome.zarr [0]",
        "scene.ome.zarr [1]",
        "scene.ome.zarr: cells",
        "tile1",
        "tile1 [1]",
    ]


def test_load_v06_scene_fused_without_coordinate_systems(
    make_napari_viewer, tmp_path
) -> None:
    """
    Without scene coordinate systems, images are fused in their intrinsic
    coordinate systems.
    """
    transforms = [
        _translation(
            [0, 0, 100],
            source={"path": "tile0", "name": "physical"},
            target={"name": "world"},
        ),
        _translation(
            [0, 0, 100],
            source={"path": "tile1", "name": "physical"},
            target={"name": "world"},
        ),
    ]
    path = _write_v06_scene(
        tmp_path / "scene.ome.zarr",
        transforms,
        ("tile0", "tile1"),
        coordinate_systems=(),
    )
    viewer = make_napari_viewer()
    with pytest.warns(UserWarning, match="Scene has no coordinate systems"):
        load_ome_zarr(
            viewer,
            zarr.open_group(path, mode="r"),
            visible=False,
            fuse_scenes=True,
        )
    image_0 = viewer.layers["scene.ome.zarr [0]"]
    np.testing.assert_allclose(image_0.translate, TRANSLATION)
    assert image_0.data[0].shape == (32, 64)


def test_load_v06_scene_fused_dtypes(make_napari_viewer, tmp_path) -> None:
    """
    Images with different dtypes are fused separately.
    """
    transforms = [
        _translation(
            [0, 0, 100 * i],
            source={"path": tile, "name": "physical"},
            target={"name": "world"},
        )
        for i, tile in enumerate(("tile0", "tile1", "tile2"))
    ]
    path = _write_v06_scene(
        tmp_path / "scene.ome.zarr",
        transforms,
        ("tile0", "tile1", "tile2"),
        dtypes={"tile1": "uint8"},
    )
    viewer = make_napari_viewer()
    load_ome_zarr(
        viewer,
        zarr.open_group(path, mode="r"),
        visible=False,
        fuse_scenes=True,
    )
    assert [layer.name for layer in viewer.layers] == [
        "scene.ome.zarr [0] (uint16)",
        "scene.ome.zarr [1] (uint16)",
        "scene.ome.zarr: cells",
        "scene.ome.zarr [0] (uint8)",
        "scene.ome.zarr [1] (uint8)",
    ]


@pytest.mark.parametrize(
    ("channel", "n_channels", "colormap"),
    [(0, 1, "gray"), (1, 2, "green"), (2, 3, "yellow"), (6, 7, "cyan")],
)
def test_channel_display(channel, n_channels, colormap) -> None:
    display = _channel_display(channel, n_channels)
    assert display["colormap"] == colormap
    assert display["blending"] == (
        "translucent_no_depth" if channel == 0 else "additive"
    )
