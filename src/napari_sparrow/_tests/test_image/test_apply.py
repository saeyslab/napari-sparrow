from spatialdata import SpatialData

from numpy.typing import NDArray
from typing import Any
from napari_sparrow.image._apply import apply, ChannelList
import pytest


def test_apply(sdata_multi_c):
    """
    Test apply on 3D image with 2 channels.
    """

    def _identity(image: NDArray, parameter: Any):
        return image

    parameter = [1, 2]
    parameter = ChannelList(parameter)

    sdata_multi_c = apply(
        sdata_multi_c,
        _identity,
        img_layer="combine_z",
        output_layer="preprocessed_apply",
        chunks=(2, 200, 200),
        channel=None,  # channel==None -> apply _identity to each layer seperately
        fn_kwargs={"parameter": parameter},
        overwrite=True,
    )

    assert "preprocessed_apply" in sdata_multi_c.images
    assert isinstance(sdata_multi_c, SpatialData)

    # test if ValueError is raised when number of parameter does not match number of
    # channels in the image.
    parameter = [1, 2, 3]
    parameter = ChannelList(parameter)

    with pytest.raises(ValueError):
        sdata_multi_c = apply(
            sdata_multi_c,
            _identity,
            img_layer="combine_z",
            output_layer="preprocessed_apply",
            chunks=(2, 200, 200),
            channel=None,  # channel==None -> apply _identity to each layer seperately
            fn_kwargs={"parameter": parameter},
            overwrite=True,
        )
