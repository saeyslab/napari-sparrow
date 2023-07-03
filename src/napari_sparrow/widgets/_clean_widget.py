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
from hydra import compose, initialize_config_dir
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from spatialdata import SpatialData
from spatialdata.transformations import Translation, set_transformation

import napari_sparrow.utils as utils
from napari_sparrow.functions import create_sdata, get_offset
from napari_sparrow.pipeline_functions import clean

log = utils.get_pylogger(__name__)

import os

from pkg_resources import resource_filename


def cleanImage(
    sdata: SpatialData,
    output_dir: str = None,
    tiling_correction_step: bool = True,
    tile_size: int = 2144,
    tophat_filtering_step: bool = True,
    size_tophat: int = None,
    clahe_processing_step: bool = True,
    contrast_clip: float = 3.5,
    chunksize_clahe: int = 20000,
    left_corner: Tuple[int, int] = None,
    size: Tuple[int, int] = None,
) -> np.ndarray:
    """Function representing the cleaning step, this calls all the needed functions to improve the image quality."""

    abs_config_dir = resource_filename("napari_sparrow", "configs")

    with initialize_config_dir(version_base=None, config_dir=abs_config_dir):
        cfg = compose(config_name="pipeline")

    cfg.paths.output_dir = output_dir

    cfg.clean.tilingCorrection = tiling_correction_step
    cfg.clean.tile_size = tile_size

    cfg.clean.tophatFiltering = tophat_filtering_step
    cfg.clean.size_tophat = size_tophat

    cfg.clean.claheProcessing = clahe_processing_step
    cfg.clean.contrast_clip = contrast_clip
    cfg.clean.chunksize_clahe = chunksize_clahe

    clean(cfg, sdata)

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
) -> list[np.ndarray]:
    """
    clean image in a thread worker
    """

    res = method(sdata, **fn_kwargs)

    return res


@magic_factory(call_button="Clean")
def clean_widget(
    viewer: napari.Viewer,
    image: napari.layers.Image,
    subset: napari.layers.Shapes,
    tiling_correction_step: bool = True,
    tile_size: int = 2144,
    tophat_filtering_step: bool = True,
    size_tophat: int = 85,
    clahe_processing_step: bool = True,
    contrast_clip: float = 3.5,
    chunksize_clahe: int = 20000,
):
    """This function represents the clean widget and is called by the wizard to create the widget."""

    if image is None:
        raise ValueError("Please select an image")

    fn_kwargs: Dict[str, Any] = {
        "output_dir": viewer.layers[utils.LOAD].metadata["output_dir"],
        "tiling_correction_step": tiling_correction_step,
        "tile_size": tile_size,
        "tophat_filtering_step": tophat_filtering_step,
        "size_tophat": size_tophat,
        "clahe_processing_step": clahe_processing_step,
        "contrast_clip": contrast_clip,
        "chunksize_clahe": chunksize_clahe,
    }

    # update this
    if len(image.data_raw.shape) == 3:
        dims = ["c", "y", "x"]
    elif len(image.data_raw.shape) == 2:
        dims = ["y", "x"]

    print(image.data_raw.shape)

    sdata = create_sdata(
        input=image.data_raw, layer_name="raw_image", chunks=1024, dims=dims
    )

    # get offset of previous layer, and set it to newly created sdata object:
    offset_x, offset_y = get_offset(
        viewer.layers[utils.LOAD].metadata["sdata"][utils.LOAD]
    )
    translation = Translation([offset_x, offset_y], axes=("x", "y"))
    set_transformation(
        sdata.images["raw_image"], translation, to_coordinate_system="global"
    )

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

        fn_kwargs["left_corner"] = left_corner
        fn_kwargs["size"] = size

    worker = _clean_worker(sdata, method=cleanImage, fn_kwargs=fn_kwargs)

    def add_image(sdata: SpatialData, layer_name: str):
        """Add the image to the napari viewer, overwrite if it already exists."""
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
            sdata[layer_name].data.squeeze(),
            translate=[
                0,
                0,
            ],
            name=layer_name,
        )

        viewer.layers[utils.CLEAN].metadata["sdata"] = sdata
        show_info("Cleaning finished")

    worker.returned.connect(lambda data: add_image(data, utils.CLEAN))
    show_info("Cleaning started")
    worker.start()
