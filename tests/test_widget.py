import pytest

from ome_zarrpari._widget import OMEZarrpariWidget


@pytest.mark.vcr
def test_example_q_widget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    # create our widget, passing in the viewer
    widget = OMEZarrpariWidget(viewer)
    assert widget.load_pane_status_text == ""
    # Load an image
    with pytest.warns(UserWarning, match="zarr array cannot be sliced lazily"):
        widget._load_ome_zarr(
            "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001247.zarr",
            visible=False,
        )
    assert widget.load_pane_status_text == "Successfully loaded"

    # Try a non-existing path
    widget._load_ome_zarr("/non/existent/path", visible=False)
    assert (
        widget.load_pane_status_text
        == "Loading OME-Zarr failed. See console for more details"
    )

    assert len(widget.added_layers) == 3

    # read captured output and check that it's as we expected
    # captured = capsys.readouterr()
    # assert captured.out == "napari has 1 layers\n"
