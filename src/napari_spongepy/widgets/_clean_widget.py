"""
Napari widget for cleaning raw (Resolve) spatial transcriptomics
microscopy images with nuclear stains. The goal of cleaning
is to improve the image quality so that subsequent image segmentation
will be more accurate.
"""

from typing import Any, Callable, Dict, Tuple

import napari
import napari.layers
import napari.types
import napari.utils
import numpy as np
import squidpy.im as sq
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info

import napari_spongepy.utils as utils

log = utils.get_pylogger(__name__)


def cleanImage(
    img: np.ndarray,
    tile_size: int = 2144,
    contrast_clip: float = 3.5,
    size_tophat: int = None,
    left_corner: Tuple[int, int] = None,
    size: Tuple[int, int] = None,
) -> np.ndarray:
    from napari_spongepy.functions import preprocessImage, tilingCorrection

    img = np.squeeze(img)
    ic = sq.ImageContainer(img)

    img, _ = tilingCorrection(ic, left_corner, size, tile_size)
    result = preprocessImage(img, contrast_clip, size_tophat)

    return result


@thread_worker(
    progress=True
)  # TODO: show string with description of current step in the napari progress bar
def _clean_worker(
    img: np.ndarray,
    method: Callable,
    fn_kwargs: Dict[str, Any],
) -> list[np.ndarray]:
    """
    clean image in a thread worker
    """

    res = method(img, **fn_kwargs)

    return res


@magic_factory(call_button="Clean")
def clean_widget(
    viewer: napari.Viewer,
    image: napari.layers.Image,
    subset: napari.layers.Shapes,
    tile_size: int = 2144,
    contrast_clip: float = 3.5,
    size_tophat: int = 85,
):

    if image is None:
        raise ValueError("Please select an image")

    fn_kwargs: Dict[str, Any] = {
        "tile_size": tile_size,
        "contrast_clip": contrast_clip,
        "size_tophat": size_tophat,
    }

    # Subset shape
    if subset:
        # Check if shapes layer only holds one shape and shape is rectangle
        if len(subset.shape_type) != 1 or subset.shape_type[0] != "rectangle":
            raise ValueError("Please select one rectangular subset")

        coordinates = np.array(subset.data[0])
        left_corner = coordinates[coordinates.sum(axis=1).argmin()].astype(int)
        size = (
            int(coordinates[:, 0].max() - coordinates[:, 0].min()),
            int(coordinates[:, 1].max() - coordinates[:, 1].min()),
        )

        fn_kwargs = {
            "tile_size": tile_size,
            "contrast_clip": contrast_clip,
            "size_tophat": size_tophat,
            "left_corner": left_corner,
            "size": size,
        }

    worker = _clean_worker(image.data, method=cleanImage, fn_kwargs=fn_kwargs)

    def add_image(img, layer_name):
        try:
            # if the layer exists, update its data
            layer = viewer.layers[layer_name]
            viewer.layers.remove(layer)

            log.info(f"Refreshing {layer_name}")
        except KeyError:
            # otherwise add it to the viewer
            log.info(f"Adding {layer_name}")

        # Translate image to appear on selected region
        viewer.add_image(
            img.data.image.squeeze().to_numpy(),
            translate=left_corner if subset else None,
            name=layer_name,
        )

        viewer.layers[utils.CLEAN].metadata["ic"] = img
        if subset:
            viewer.layers[utils.CLEAN].metadata["left_corner"] = left_corner
        show_info("Cleaning finished")

    worker.returned.connect(lambda data: add_image(data, utils.CLEAN))
    show_info("Cleaning started")
    worker.start()
