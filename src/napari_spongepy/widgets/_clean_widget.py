"""
Napari widget for cleaning raw (Resolve) spatial transcriptomics
microscopy images with nuclear stains. The goal of cleaning
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
from basicpy import BaSiC
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from scipy import ndimage
from squidpy.im import ImageContainer

import napari_spongepy.utils as utils


def BasiCCorrection(img: np.ndarray) -> np.ndarray:
    "This function corrects for the tiling effect that occurs in RESOLVE data"
    basic = BaSiC(get_darkfield=True, lambda_flatfield_coef=10, device="cpu")
    basic.fit(img)
    tiles_corrected = basic.transform(img)
    return basic.transform(tiles_corrected)


def cleanImage(
    img: np.ndarray,
    contrast_clip: float = 2.5,
    size_tophat: int = None,
) -> np.ndarray:

    img = np.squeeze(img)

    # mask black lines
    mask_lines = np.where(img == 0)  # find the location of the lines
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask[mask_lines[0], mask_lines[1]] = 1  # put one values in the correct position

    # perform inpainting
    # res_ns = cv2.inpaint(img, mask, 55, cv2.INPAINT_NS)
    # img = res_ns

    # tophat filter
    if size_tophat is not None:
        minimum_t = ndimage.minimum_filter(img, size_tophat)
        max_of_min_t = ndimage.maximum_filter(minimum_t, size_tophat)
        orig_sub_min = img - max_of_min_t
        img = orig_sub_min

    # enhance contrast
    clahe = cv2.createCLAHE(clipLimit=contrast_clip, tileGridSize=(8, 8))
    img = clahe.apply(img)
    return img


@thread_worker(
    progress=True
)  # TODO: show string with description of current step in the napari progress bar
def _clean_worker(
    img, tophat_size: int, contrast_clip: float
) -> Generator[napari.types.LayerDataTuple, None, None]:
    """
    clean image in a thread worker
    """
    ic = ImageContainer(img, label="image", chunks="auto")
    ic = ic.apply(
        cleanImage,
        new_layer="cleanImage",
        chunks=(1_000, 1_000, 1),
        channel=0,
        lazy=True,
        contrast_clip=contrast_clip,
        size_tophat=tophat_size,
    )
    # ic = ic["cleanImage"].apply(cleanImage, contrast_clip=contrast_clip, size_tophat=tophat_size, name="cleanImage")
    return utils.ic_to_da(ic, label="cleanImage")


@magic_factory(call_button="clean")
def clean_widget(
    viewer: napari.Viewer,
    image: napari.layers.Image,
    tophat_size: int = 45,
    contrast_clip: float = 2.5,
) -> None:
    print(
        f"About to clean {image}; tophat_size={tophat_size} contrast_clip={contrast_clip}"
    )
    if image is None:
        return

    worker = _clean_worker(image.data, tophat_size, contrast_clip)
    worker.returned.connect(lambda data: _update_layer(viewer, data, "cleanImage"))
    worker.start()


def _update_layer(viewer, img, name):
    try:
        # if the layer exists, update its data
        viewer.layers[name].data = img
    except KeyError:
        # otherwise add it to the viewer
        viewer.add_image(img, name=name)


if __name__ == "__main__":
    from skimage import io

    img = io.imread("data/resolve_liver/20272_slide1_A1-1_DAPI.tiff")
    ic = ImageContainer(img)
    ic = ic.apply(
        cleanImage,
        new_layer="cleanImage",
        lazy=True,
        contrast_clip=45,
        size_tophat=2.5,
    )
    ic["cleanImage"].data
