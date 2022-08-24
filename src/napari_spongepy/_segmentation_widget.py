"""
Napari widget for cell segmentation of
preprocessed (Resolve) spatial transcriptomics
microscopy images with nuclear stains.
Segmentation is performed with Squidpy ImageContainer and segment.
Setting "Enable async tiling" is needed to see intermediate results.
> The tiles do not seem to be computed
Setting "Render Images async" is needed to remove jank from rendering.
> However, this messes with the cache and needlessly does segmentation on movement
"""

from enum import Enum
from typing import Callable

import dask.array as da
import napari
import napari.layers
import napari.types
import numpy as np
import toolz as tz
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from squidpy.im import ImageContainer, segment

import napari_spongepy.utils as utils

log = utils.get_pylogger(__name__)


class SegmentationOption(Enum):
    watershed = "watershed"
    cellpose = "cellpose"
    # log = "log"
    # dog = "dog"
    # doh = "doh"


@tz.curry
def create_cellpose_method(device: str):
    from cellpose import models

    # Needs to be recreated, else AttributeError: 'CPnet' object has no attribute 'diam_mean'
    model = models.Cellpose(device=device, model_type="nuclei")

    def cellpose_method(img, fn_kwargs: dict):
        log.info(f"segmenting {img.shape}")
        masks, _, _, _ = model.eval(
            img,
            channels=[0, 0],
            **fn_kwargs,
        )
        return masks

    return cellpose_method


def toggle_layer_vis_on_zoom(viewer, layer_name, zoom_threshold):
    """
    Make a layer visible or invisible when the viewer is zoomed in or out.
    Invisible layers are not rendered or computed.
    Catch KeyError should layer not be in the LayerList.
    """

    def f(event, layer_name=layer_name, zoom_threshold=zoom_threshold):
        try:
            layer = viewer.layers[layer_name]
            layer.visible = viewer.camera.zoom > zoom_threshold
        except KeyError:
            pass

    return f


@thread_worker
def _segmentation_worker(
    ic: np.ndarray | ImageContainer,
    method: Callable | str,
    subset=None,
    reduce_z=None,
    reduce_c=None,
    fn_kwargs=None,
    # if async interactive works: smaller chunks for faster segmentation computation
    chunks=(1000, 1000, 1, 1),
) -> list[np.ndarray]:

    label_image = "image"
    label_segmentation = "segment_watershed"
    if not isinstance(ic, ImageContainer):
        ic = utils.get_ic(img=ic, label=label_image, chunks=chunks)
    segment(
        ic,
        layer="image",
        method=method,
        layer_added=label_segmentation,
        lazy=True,
        chunks=chunks,
        fn_kwargs=fn_kwargs,
    )
    s = utils.ic_to_da(ic, "segment_watershed", reduce_c=reduce_c, reduce_z=reduce_z)
    if subset:
        s = s[subset]

    # make a dummy lower-res array to trigger multi-scale rendering
    dummy_s = da.zeros(tuple(np.array(s.shape) // 2)).astype(np.uint8)
    ss = [s, dummy_s]
    return ss


@magic_factory(call_button="Segment")
def segmentation_widget(
    viewer: napari.Viewer,
    image: napari.layers.Image,
    method: SegmentationOption = SegmentationOption.watershed,
    device: str = "cpu",
    min_size: int = 80,
    flow_threshold: float = 0.6,
    diameter: int = 55,
    cellprob_threshold: int = 0,
    thresh: int = 10_000,
    geq: bool = True,
) -> None:

    log.info(f"About to segment {image} using {method}; device={device}")
    if image is None:
        return
    if method == SegmentationOption.cellpose:
        method_fn = create_cellpose_method(device)
        fn_kwargs = {
            "min_size": min_size,
            "flow_threshold": flow_threshold,
            "diameter": diameter,
            "cellprob_threshold": cellprob_threshold,
            "resample": False,
        }
    else:
        method_fn = method.value
        fn_kwargs = {
            "thresh": thresh,
            "geq": geq,
        }
    worker = _segmentation_worker(image.data, method_fn, fn_kwargs=fn_kwargs)
    log.info("Worker created")

    layer_name = "Segmentation"

    def add_labels(img):
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
        viewer.add_labels(img, visible=True, name=layer_name)
        # f = toggle_layer_vis_on_zoom(viewer, layer_name, zoom_threshold=0.9)
        # viewer.camera.events.zoom.connect(f)
        # execute f to emulate zoom event and set visiblity correct
        # f(None)
        return viewer

    worker.returned.connect(add_labels)
    # worker.returned.connect(
    #     lambda label_img: viewer.add_labels(label_img, name="Segmentation")
    # )
    worker.start()
