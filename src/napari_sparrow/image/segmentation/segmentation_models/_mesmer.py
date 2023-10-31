from typing import Optional

import numpy as np
from numpy.typing import NDArray

try:
    from deepcell.applications import Mesmer

    DEEPCELL_AVAILABLE = True
except ImportError:
    DEEPCELL_AVAILABLE = False

if DEEPCELL_AVAILABLE:
    # download model weights from https://deepcell-data.s3-us-west-1.amazonaws.com/saved-models/MultiplexSegmentation-9.tar.gz
    app = Mesmer()
else:
    raise RuntimeError(
        "The deepcell module is not available. Please install it to use this function."
    )


def _mesmer(
    img: NDArray,
    image_mpp: int = 0.2,
    nuc_channel: Optional[int] = None,
    mem_channel: Optional[int] = None,
    compartment: str = "whole-cell",
) -> NDArray:
    if nuc_channel is None and mem_channel is None:
        raise ValueError("Please specify either nuc_channel, mem_channel or both.")

    if nuc_channel is None or mem_channel is None:
        specified_channel = nuc_channel if nuc_channel is not None else mem_channel
        channel_img = img[..., specified_channel][..., None]
        zeros_arr = np.zeros((img.shape[0], img.shape[1], 1))
        img = np.concatenate((channel_img, zeros_arr), axis=-1)

    # If both channels are specified
    else:
        nuc_img = img[..., nuc_channel][..., None]
        mem_img = img[..., mem_channel][..., None]
        img = np.concatenate((nuc_img, mem_img), axis=-1)

    # mesmer want img to be of dimension (batch, y, x, c)
    # we give it input (z, y, x, c), with z dim ==1.
    masks = app.predict(img, image_mpp=image_mpp, compartment=compartment)

    # mesmer returns dimension (batch, y,x,c), with in our use case batch==1, and c==1.
    # so it is in correct format (z,y,x,c)
    return masks
