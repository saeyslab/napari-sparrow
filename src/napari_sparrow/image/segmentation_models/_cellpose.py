from typing import List
from numpy.typing import NDArray

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from cellpose import models

    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False


def _cellpose(
    img: NDArray,
    min_size: int = 80,
    cellprob_threshold: int = 0,
    flow_threshold: float = 0.6,
    diameter: int = 55,
    model_type: str = "nuclei",
    channels: List[int] = [0, 0],
    device: str = "cpu",
) -> NDArray:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "The torch module is not available. Please install it to use this function."
        )

    if not CELLPOSE_AVAILABLE:
        raise RuntimeError(
            "The cellpose module is not available. Please install it to use this function."
        )

    gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    model = models.Cellpose(gpu=gpu, model_type=model_type, device=torch.device(device))
    masks, _, _, _ = model.eval(
        img,
        diameter=diameter,
        channels=channels,
        min_size=min_size,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    return masks
