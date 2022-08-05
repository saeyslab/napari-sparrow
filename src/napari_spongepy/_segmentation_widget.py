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


@tz.curry
def create_cellpose_method(
    use_gpu,
    min_size: int,
    flow_threshold: float,
    diameter: int,
    mask_threshold: int,
):
    from cellpose import models

    channels = np.array([0, 0])

    def cellpose_method(img):
        # Needs to be recreated, else AttributeError: 'CPnet' object has no attribute 'diam_mean'
        log.info(f"segmenting {img.shape}")
        model = models.Cellpose(gpu=use_gpu, model_type="nuclei")
        masks, _, _, _ = model.eval(
            img,
            diameter=diameter,
            channels=channels,
            min_size=min_size,
            flow_threshold=flow_threshold,
            cellprob_threshold=mask_threshold,
        )
        return masks

    return cellpose_method


def ic_to_da(ic, label="image", drop_dims=["z", "channels"]):
    """
    ImageContainer defaults to (x, y, z, channels), most of the time we need just (x, y)
    """
    return ic[label].squeeze(dim=drop_dims).data


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
    img: np.ndarray,
    method: str,
    subset=None,
    chunks="auto",
    # if async interactive works: smaller chunks for faster segmentation computation
    # chunks = (500, 500, 1, 1),
) -> list[np.ndarray]:

    label_image = "image"
    label_segmentation = "segment_watershed"
    ic = ImageContainer(img, label=label_image, chunks=chunks)
    segment(
        ic,
        layer="image",
        method=method,
        layer_added=label_segmentation,
        lazy=True,
        chunks=chunks,
    )
    s = ic_to_da(ic, "segment_watershed")
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
    method: str = "watershed",
    use_gpu: bool = False,
    min_size: int = 80,
    flow_threshold: float = 0.6,
    diameter: int = 55,
    mask_threshold: int = 0,
) -> None:

    log.info(f"About to segment {image} using {method}; use_gpu={use_gpu}")
    if image is None:
        return

    if method == "cellpose":
        method = create_cellpose_method(
            use_gpu=use_gpu,
            min_size=min_size,
            flow_threshold=flow_threshold,
            diameter=diameter,
            mask_threshold=mask_threshold,
        )

    worker = _segmentation_worker(image.data, method)

    layer_name = "Segmentation"

    def add_labels(img):
        viewer.add_image(img, visible=False, name=layer_name)
        f = toggle_layer_vis_on_zoom(viewer, layer_name, zoom_threshold=0.9)
        viewer.camera.events.zoom.connect(f)
        # execute f to emulate zoom event and set visiblity correct
        f(None)
        return viewer

    worker.returned.connect(add_labels)
    # worker.returned.connect(
    #     lambda label_img: viewer.add_labels(label_img, name="Segmentation")
    # )
    worker.start()
