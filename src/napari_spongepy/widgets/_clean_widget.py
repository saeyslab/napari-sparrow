"""
Napari widget for cleaning raw (Resolve) spatial transcriptomics
microscopy images with nuclear stains. The goal of cleaning
is to improve the image quality so that subsequent image segmentation
will be more accurate.
"""
from typing import Callable

import napari
import napari.layers
import napari.types
import napari.utils
import numpy as np
from magicgui import magic_factory
from napari.qt.threading import thread_worker

import napari_spongepy.utils as utils

log = utils.get_pylogger(__name__)


def cleanImage(
    img: np.ndarray,
    contrast_clip: float = 3.5,
    size_tophat: int = None,
) -> np.ndarray:
    from napari_spongepy.functions import preprocessImage, tilingCorrection

    img = np.squeeze(img)
    img = img[:4288, :4288]

    img, _ = tilingCorrection(img)

    result = preprocessImage(img, contrast_clip, size_tophat)

    return result


@thread_worker(
    progress=True
)  # TODO: show string with description of current step in the napari progress bar
def _clean_worker(
    img: np.ndarray,
    method: Callable,
    subset=None,
    fn_kwargs=None,
) -> list[np.ndarray]:
    """
    clean image in a thread worker
    """

    res = method(img, **fn_kwargs)

    return res


@magic_factory(call_button="Clean", result_widget=True)
def clean_widget(
    viewer: napari.Viewer,
    image: napari.layers.Image,
    size_tophat: int = 85,
    contrast_clip: float = 3.5,
) -> str:
    log.info(
        f"About to clean {image}; size_tophat={size_tophat} contrast_clip={contrast_clip}"
    )
    if image is None:
        return "Please select an image"

    fn_kwargs = {
        "contrast_clip": contrast_clip,
        "size_tophat": size_tophat,
    }

    worker = _clean_worker(image.data, method=cleanImage, fn_kwargs=fn_kwargs)
    log.info("Cleaning worker created")

    def add_image(img, layer_name):
        try:
            # if the layer exists, update its data
            layer = viewer.layers[layer_name]
            viewer.layers.remove(layer)

            log.info(f"Refreshing {layer_name}")
            # layer.refresh()
        except KeyError:
            # otherwise add it to the viewer
            log.info(f"Adding {layer_name}")

        viewer.add_image(img, name=layer_name)
        log.info("Cleaning finished")

        return "Cleaning finished"

    worker.returned.connect(lambda data: add_image(data, utils.CLEAN))
    worker.start()
    return "Cleaning started"
