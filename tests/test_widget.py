import numpy as np
import pytest

from ome_zarrpari._widget import OMEZarrpariWidget


@pytest.mark.vcr
def test_example_q_widget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))

    # create our widget, passing in the viewer
    widget = OMEZarrpariWidget(viewer)
    # Load an image
    widget._load_ome_zarr(
        "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0066/ExpD_chicken_embryo_MIP.ome.zarr",
        visible=False,
    )

    # read captured output and check that it's as we expected
    # captured = capsys.readouterr()
    # assert captured.out == "napari has 1 layers\n"
