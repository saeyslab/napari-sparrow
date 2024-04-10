from __future__ import annotations

from pathlib import Path

import torch
from numpy.typing import NDArray

from sparrow.utils.pylogger import get_pylogger

# try:
#    import torch

#    TORCH_AVAILABLE = True
# except ImportError:
#   TORCH_AVAILABLE = False

try:
    from cellpose import models

    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False

log = get_pylogger(__name__)


def _cellpose(
    img: NDArray,
    min_size: int = 80,
    cellprob_threshold: int = 0,
    flow_threshold: float = 0.6,
    diameter: int = 55,
    model_type: str = "nuclei",
    pretrained_model: str | Path | None = None,
    channels: list[int] | None = None,
    device: str = "cpu",
    z_axis: int = 0,
    channel_axis: int = 3,
    do_3D: bool = False,
    anisotropy: float = 2,
) -> NDArray:
    if channels is None:
        channels = [0, 0]
    if not CELLPOSE_AVAILABLE:
        raise RuntimeError("The cellpose module is not available. Please install it to use this function.")
    gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    if pretrained_model is not None:
        model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model, device=torch.device(device))
    elif model_type is not None:
        model = models.Cellpose(gpu=gpu, model_type=model_type, device=torch.device(device))
    else:
        raise ValueError(
            "Please provide either 'model_type' or 'pretrained_model (i.e. a path to a pretrained model)'."
        )
    results = model.eval(
        img,
        diameter=diameter,
        channels=channels,
        min_size=min_size,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        z_axis=z_axis,
        channel_axis=channel_axis,
        do_3D=do_3D,
        anisotropy=anisotropy,
    )

    masks = results[0]

    # make sure we always return z,y,x for labels.
    if not do_3D:
        masks = masks[None, ...]

    # add trivial channel dimension, so we return z,y,x,c
    masks = masks[..., None]

    return masks
