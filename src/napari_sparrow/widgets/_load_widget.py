"""
Napari widget for cleaning raw (Resolve) spatial transcriptomics
microscopy images with nuclear stains. The goal of cleaning
is to improve the image quality so that subsequent image segmentation
will be more accurate.
"""

from typing import Any, Callable, Dict, List, Optional

import napari
import napari.layers
import napari.types
import napari.utils
import numpy as np
import squidpy.im as sq
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
import os

import napari_sparrow.utils as utils
from napari_sparrow.functions import (
    create_sdata,
    get_offset,
)

from spatialdata import SpatialData
from pathlib import Path

log = utils.get_pylogger(__name__)


def loadImage(
    sdata: SpatialData,
) -> SpatialData:
    """Function representing the cleaning step, this calls all the needed functions to improve the image quality."""

    return sdata


@thread_worker(
    progress=True
)  # TODO: show string with description of current step in the napari progress bar
def _load_worker(
    sdata: SpatialData,
    method: Callable,
    fn_kwargs: Dict[str, Any],
) -> list[np.ndarray]:
    """
    clean image in a thread worker
    """

    res = method(sdata, **fn_kwargs)

    return res


@magic_factory(
    call_button="Load",
    path_image={"widget_type": "FileEdit"},
    output_dir={"widget_type": "FileEdit", "mode": "d"},
)
def load_widget(
    viewer: napari.Viewer,
    path_image: Path = Path(""),
    output_dir: Path = Path(""),
    x_min: Optional[str] = "-",
    x_max: Optional[str] = "-",
    y_min: Optional[str] = "-",
    y_max: Optional[str] = "-",
):
    """This function represents the clean widget and is called by the wizard to create the widget."""



    crd=[x_min, x_max, y_min, y_max]
    crd = [None if val == "-" else int(val) for val in crd]

    show_info("Creating SpatialData object")
    if path_image:
        sdata = create_sdata(
            input=path_image,
            output_path=os.path.join(output_dir, 'sdata.zarr' ),
            layer_name=utils.LOAD,
            chunks=1024,
            crd=crd if crd else None,
        )

    # elif image:
    # need to pass this as arguments
    #    if len( image.data_raw.shape )==3:
    #        dims=[ 'y', 'x', 'c' ]
    #    elif len( image.data_raw.shape )==2:
    #        dims=[ 'y', 'x' ]
    #    sdata=create_sdata( input=image.data_raw, layer_name=utils.LOAD, chunks=1024, dims=dims )
    else:
        raise ValueError("Please select an image, or set a path to an image")


    fn_kwargs: Dict[str, Any] = {}

    worker = _load_worker(sdata, method=loadImage, fn_kwargs=fn_kwargs)

    def add_image(sdata: SpatialData, layer_name: str, output_dir: str):
        """Add the image to the napari viewer, overwrite if it already exists."""
        try:
            # if the layer exists, update its data
            layer = viewer.layers[layer_name]
            viewer.layers.remove(layer)

            log.info(f"Refreshing {layer_name}")
        except KeyError:
            # otherwise add it to the viewer
            log.info(f"Adding {layer_name}")

        offset_x, offset_y = get_offset(sdata[layer_name])

        # Translate image to appear on selected region
        viewer.add_image(
            sdata[layer_name].data.squeeze(),
            translate=[
                offset_y,
                offset_x,
            ],
            name=layer_name,
        )

        viewer.layers[utils.LOAD].metadata["sdata"] = sdata
        viewer.layers[utils.LOAD].metadata["output_dir"] = output_dir
        show_info("Loading finished")

    worker.returned.connect(lambda data: add_image(data, utils.LOAD, output_dir ))
    show_info("Loading started")
    worker.start()
