"""
Napari widget for preprocessing raw (Resolve) spatial transcriptomics
microscopy images with nuclear stains. The goal of preprocessing
is to improve the image quality so that subsequent image segmentation
will be more accurate.
"""
from typing import Generator

import cv2
import napari
import napari.layers
import napari.types
import napari.utils
import numpy as np
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from scipy import ndimage


@thread_worker(
    progress=True
)  # TODO: show string with description of current step in the napari progress bar
def _preprocess_worker(
    img, tophat_size: int, contrast_clip: float
) -> Generator[napari.types.LayerDataTuple, None, None]:

    # Find mask for inpainting the black tile boundary lines in raw Resolve images.
    maskLines = np.where(img == 0)  # find the location of the lines
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask[maskLines[0], maskLines[1]] = 1  # put one values in the correct position
    yield (
        mask,
        {"name": "Missing pixels"},
    )

    # Perform Navier-Stokes inpainting on the black tile boundary lines.
    inpainted_img = cv2.inpaint(
        img, inpaintMask=mask, inpaintRadius=15, flags=cv2.INPAINT_NS
    )
    img = inpainted_img
    yield (
        inpainted_img,
        {"name": "Inpainted"},
    )

    # Remove background using a tophat filter.
    local_minimum_img = ndimage.minimum_filter(img, tophat_size)
    tophat_filtered_img = img - local_minimum_img
    img = tophat_filtered_img
    yield (
        tophat_filtered_img,
        {"name": "Tophat filtered"},
    )

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=contrast_clip, tileGridSize=(8, 8))
    img = clahe.apply(img)
    yield (
        img,
        {"name": "Preprocessed"},
    )


@magic_factory(call_button="Preprocess")
def preprocess_widget(
    viewer: napari.Viewer,
    image: napari.layers.Image,
    tophat_size: int = 45,
    contrast_clip: float = 2.5,
) -> None:
    print(
        f"About to preprocess {image}; tophat_size={tophat_size} contrast_clip={contrast_clip}"
    )
    if image is None:
        return

    worker = _preprocess_worker(image.data, tophat_size, contrast_clip)
    worker.yielded.connect(lambda ldt: _update_layer(viewer, ldt[0], ldt[1]["name"]))
    worker.start()


def _update_layer(viewer, img, name):
    try:
        # if the layer exists, update its data
        viewer.layers[name].data = img
    except KeyError:
        # otherwise add it to the viewer
        viewer.add_image(img, name=name)
