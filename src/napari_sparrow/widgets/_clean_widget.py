"""
Napari widget for cleaning raw (Resolve) spatial transcriptomics
microscopy images with nuclear stains. The goal of cleaning
is to improve the image quality so that subsequent image segmentation
will be more accurate.
"""

from typing import Any, Callable, Dict, Optional, List
import os

import napari
import napari.layers
import napari.types
import napari.utils
import numpy as np
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from omegaconf.dictconfig import DictConfig
from spatialdata import SpatialData
from spatialdata.transformations import Translation, set_transformation

from napari_sparrow.pipeline import clean
from napari_sparrow import utils as utils
from napari_sparrow.image._image import _get_translation
from napari_sparrow.io import create_sdata

log = utils.get_pylogger(__name__)


def cleanImage(
    sdata: SpatialData,
    cfg: DictConfig,
) -> SpatialData:
    """Function representing the cleaning step, this calls all the needed functions to improve the image quality."""

    sdata = clean(cfg, sdata)

    # add the last layer as the CLEAN layer
    layer_name = utils.CLEAN
    layer = [*sdata.images][-1]

    sdata.add_image(name=layer_name, image=sdata[layer])

    return sdata


@thread_worker(
    progress=True
)  # TODO: show string with description of current step in the napari progress bar
def _clean_worker(
    sdata: SpatialData,
    method: Callable,
    fn_kwargs: Dict[str, Any],
) -> SpatialData:
    """
    clean image in a thread worker
    """

    res = method(sdata, **fn_kwargs)

    return res


@magic_factory(call_button="Clean")
def clean_widget(
    viewer: napari.Viewer,
    image: napari.layers.Image,
    subset: Optional[napari.layers.Shapes] = None,
    tiling_correction_step: bool = True,
    tile_size: int = 2144,
    min_max_filtering_step: bool = True,
    size_min_max_filter: List[int] = [85],
    contrast_enhancing_step: bool = True,
    contrast_clip: List[float] = [3.5],
):
    """This function represents the clean widget and is called by the wizard to create the widget."""

    if image is None:
        raise ValueError("Please select an image")

    cfg = viewer.layers[utils.LOAD].metadata["cfg"]

    cfg.clean.tilingCorrection = tiling_correction_step
    cfg.clean.tile_size = tile_size

    cfg.clean.minmaxFiltering = min_max_filtering_step
    cfg.clean.size_min_max_filter = size_min_max_filter

    cfg.clean.contrastEnhancing = contrast_enhancing_step
    cfg.clean.contrast_clip = contrast_clip

    # update this
    if len(image.data_raw.shape) == 3:
        dims = ["c", "y", "x"]
    elif len(image.data_raw.shape) == 2:
        dims = ["y", "x"]

    if image.name == utils.LOAD:
        # We need to create new sdata object, because sdata object in
        # viewer.layers[utils.LOAD].metadata["sdata"][utils.LOAD] is backed by .zarr store
        # and we are not allowed to overwrite it (i.e. we would not be allowed to run the cleaning step twice)
        sdata = create_sdata(
            input=image.data_raw, img_layer="raw_image", chunks=1024, dims=dims
        )

        if 'sdata' in viewer.layers[utils.LOAD].metadata:
            # get offset of previous layer, and set it to newly created sdata object:
            offset_x, offset_y = _get_translation(
                viewer.layers[utils.LOAD].metadata["sdata"][utils.LOAD]
            )
            translation = Translation([offset_x, offset_y], axes=("x", "y"))
            set_transformation(
                sdata.images["raw_image"], translation, to_coordinate_system="global"
            )

    else:
        raise ValueError(
            f"Please run the cleaning step on the layer with name '{utils.LOAD}',"
            f"it seems layer with name '{image.name}' was selected."
        )

    if subset:
        # Check if shapes layer only holds one shape and shape is rectangle
        if len(subset.shape_type) != 1 or subset.shape_type[0] != "rectangle":
            raise ValueError("Please select one rectangular subset")

        coordinates = np.array(subset.data[0])
        crd = [
            int(coordinates[:, 1].min()),
            int(coordinates[:, 1].max()),
            int(coordinates[:, 0].min()),
            int(coordinates[:, 0].max()),
        ]

        # FIXME note crd will be ignored if you do not do tiling correction
        cfg.clean.crop_param = crd

    else:
        cfg.clean.crop_param = None

    cfg.clean.small_size_vis=cfg.clean.crop_param

    fn_kwargs: Dict[str, Any] = {
        "cfg": cfg,
    }

    worker = _clean_worker(sdata, method=cleanImage, fn_kwargs=fn_kwargs)

    def add_image(sdata: SpatialData, cfg: DictConfig, layer_name: str):
        """Add the image to the napari viewer, overwrite if it already exists."""
        try:
            # if the layer exists, update its data
            layer = viewer.layers[layer_name]
            viewer.layers.remove(layer)

            log.info(f"Refreshing {layer_name}")
        except KeyError:
            # otherwise add it to the viewer
            log.info(f"Adding {layer_name}")

        offset_x, offset_y = _get_translation(sdata[layer_name])

        # Translate image to appear on selected region
        viewer.add_image(
            sdata[layer_name].data.squeeze(),
            translate=[
                offset_y,
                offset_x,
            ],
            name=layer_name,
        )

        viewer.layers[layer_name].metadata["sdata"] = sdata
        viewer.layers[layer_name].metadata["cfg"] = cfg

        log.info( f"Added {utils.CLEAN} layer" )

        utils._export_config( cfg.clean, os.path.join( cfg.paths.output_dir, 'configs', 'clean', 'plugin.yaml' ) )
        log.info( "Cleaning finished" )
        show_info("Cleaning finished")

    worker.returned.connect(lambda data: add_image(data, cfg, utils.CLEAN))
    log.info( "Cleaning started" )
    show_info("Cleaning started")
    worker.start()

    # return worker, so status can be checked for unit testing
    return worker