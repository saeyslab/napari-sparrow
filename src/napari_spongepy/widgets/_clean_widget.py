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


def BasiCCorrection(img: np.ndarray) -> np.ndarray:
    from basicpy import BaSiC

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
    from napari_spongepy.functions import preprocessImage

    img = np.squeeze(img)

    result, _ = preprocessImage(img)

    result = result[:, :, np.newaxis, np.newaxis]
    return result


@thread_worker(
    progress=True
)  # TODO: show string with description of current step in the napari progress bar
def _clean_worker(
    img: np.ndarray,
    method: Callable,
    subset=None,
    fn_kwargs=None,
    reduce_z=None,
    reduce_c=None,
    # if async interactive works: smaller chunks for faster segmentation computation
    # chunks=(1000, 1000, 1, 1),
    chunks="auto",
) -> list[np.ndarray]:
    """
    clean image in a thread worker
    """

    ic = utils.get_ic(img, layer=utils.IMAGE)
    new_ic = ic.apply(
        func=method,
        layer=utils.IMAGE,
        new_layer=utils.CLEAN,
        lazy=True,
        copy=True,
        chunks=chunks,
        fn_kwargs=fn_kwargs,
    )
    ic.add_img(new_ic, layer=utils.CLEAN, lazy=True)
    s = utils.ic_to_da(ic, utils.CLEAN, reduce_c=reduce_c, reduce_z=reduce_z)
    # make a dummy lower-res array to trigger multi-scale rendering
    # dummy_s = da.zeros(tuple(np.array(s.shape) // 2)).astype(np.uint8)
    # ss = [s, dummy_s]
    return s


# def layer_options(gui) -> list[tuple[str, Any]]:
#     # `gui` will be the combobox instance... if you want, you can traverse
#     # the widget parent chain to find the viewer and do whatever you want.
#     # but here, we just return a simple list of strings that we want
#     # the combobox to show
#     from napari.utils._magicgui import get_layers_data

#     layers_data = get_layers_data(gui)

#     try:
#         index = [x[0] for x in layers_data].index(utils.CLEAN)
#     except ValueError:
#         index = None
#     if index:
#         l = layers_data.pop(index)
#         return l + layers_data
#     return layers_data


@magic_factory(call_button="clean")
def clean_widget(
    viewer: napari.Viewer,
    image: napari.layers.Image,
    size_tophat: int = 45,
    contrast_clip: float = 2.5,
) -> None:
    log.info(
        f"About to clean {image}; size_tophat={size_tophat} contrast_clip={contrast_clip}"
    )
    if image is None:
        return

    fn_kwargs = {
        "contrast_clip": contrast_clip,
        "size_tophat": size_tophat,
    }

    worker = _clean_worker(image.data, method=cleanImage, fn_kwargs=fn_kwargs)
    worker.returned.connect(lambda data: add_image(data, utils.CLEAN))
    worker.start()
    log.info("Worker created")

    def add_image(img, layer_name):
        try:
            # if the layer exists, update its data
            layer = viewer.layers[layer_name]
            viewer.layers.remove(layer)

            # layer.data = img
            log.info(f"Refreshing {layer_name}")
            # layer.refresh()
        except KeyError:
            # otherwise add it to the viewer
            log.info(f"Adding {layer_name}")
        viewer.add_image(img, name=layer_name)
        # f = toggle_layer_vis_on_zoom(viewer, layer_name, zoom_threshold=0.9)
        # viewer.camera.events.zoom.connect(f)
        # execute f to emulate zoom event and set visiblity correct
        # f(None)
        log.debug(print(utils.get_ic()))
        return viewer


if __name__ == "__main__":
    from skimage import io
    from squidpy.im import ImageContainer

    img = io.imread("data/resolve_liver/20272_slide1_A1-1_DAPI.tiff")
    ic = ImageContainer(img, layer=utils.CLEAN)
    fn_kwargs = {
        "contrast_clip": 45,
        "size_tophat": 2.5,
    }
    ic = ic.apply(
        func=cleanImage,
        layer=utils.IMAGE,
        new_layer=utils.CLEAN,
        lazy=False,
        chunks="auto",
        fn_kwargs=fn_kwargs,
    )
    ic[utils.CLEAN].data
    s = utils.ic_to_da(ic, utils.CLEAN, reduce_c=False, reduce_z=False)
    s
