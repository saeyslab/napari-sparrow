"""
Napari widget for cell segmentation of
cleaned (Resolve) spatial transcriptomics
microscopy images with nuclear stains.
Segmentation is performed with Squidpy ImageContainer and segment.
Setting "Enable async tiling" is needed to see intermediate results.
> The tiles do not seem to be computed
Setting "Render Images async" is needed to remove jank from rendering.
> However, this messes with the cache and needlessly does segmentation on movement
"""

from enum import Enum
from typing import Callable, List

import napari
import napari.layers
import napari.types
import numpy as np
from magicgui import magic_factory
from napari.qt.threading import thread_worker

import napari_spongepy.utils as utils

log = utils.get_pylogger(__name__)


class ModelOption(Enum):
    nuclei = "nuclei"
    cyto = "cyto"


def segmentImage(
    img: np.ndarray,
    device: str = "cpu",
    min_size: int = 80,
    flow_threshold: float = 0.6,
    diameter: int = 55,
    cellprob_threshold: int = 0,
    model_type: str = "nuclei",
    channels: List[int] = [0, 0],
) -> np.ndarray:
    from napari_spongepy.functions import segmentation

    img = np.squeeze(img)

    mask, _, _ = segmentation(
        img,
        device,
        min_size,
        flow_threshold,
        diameter,
        cellprob_threshold,
        model_type,
        channels,
    )

    return mask


@thread_worker(progress=True)
def _segmentation_worker(
    img: np.ndarray,
    method: Callable,
    fn_kwargs=None,
) -> list[np.ndarray]:
    res = method(img, **fn_kwargs)

    return res


@magic_factory(call_button="Segment")
def segment_widget(
    viewer: napari.Viewer,
    image: napari.layers.Image,
    device: str = "cpu",
    min_size: int = 80,
    flow_threshold: float = 0.8,
    diameter: int = 50,
    cellprob_threshold: int = -2,
    model_type: ModelOption = ModelOption.nuclei,
) -> None:

    log.info(f"About to segment {image} using cellpose; device={device}")
    if image is None:
        return
    else:
        method_fn = segmentImage
        fn_kwargs = {
            "device": device,
            "min_size": min_size,
            "flow_threshold": flow_threshold,
            "diameter": diameter,
            "cellprob_threshold": cellprob_threshold,
            "model_type": model_type.value,
            "channels": [0, 0],
        }
    worker = _segmentation_worker(image.data, method_fn, fn_kwargs=fn_kwargs)
    log.info("Worker created")

    layer_name = utils.SEGMENT

    def add_labels(img):
        try:
            # if the layer exists, update its data
            layer = viewer.layers[layer_name]
            viewer.layers.remove(layer)

            log.info(f"Refreshing {layer_name}")
        except KeyError:
            # otherwise add it to the viewer
            log.info(f"Adding {layer_name}")
        viewer.add_labels(img, visible=True, name=layer_name)

        return viewer

    worker.returned.connect(add_labels)
    worker.start()
