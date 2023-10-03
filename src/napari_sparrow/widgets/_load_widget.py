"""
Napari widget for cleaning raw (Resolve) spatial transcriptomics
microscopy images with nuclear stains. The goal of cleaning
is to improve the image quality so that subsequent image segmentation
will be more accurate.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import napari
import napari.layers
import napari.types
import napari.utils
import numpy as np
import squidpy.im as sq
from hydra import compose, initialize_config_dir
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from pkg_resources import resource_filename
from spatialdata import SpatialData

import napari_sparrow.utils as utils
from napari_sparrow.image._image import _get_translation
from napari_sparrow.pipeline import SparrowPipeline

log = utils.get_pylogger(__name__)


def loadImage(
    pipeline: SparrowPipeline,
) -> SpatialData:
    """Function representing the loading step."""

    sdata = pipeline.load()

    return sdata


@thread_worker(
    progress=True
)  # TODO: show string with description of current step in the napari progress bar
def _load_worker(
    method: Callable,
    fn_kwargs: Dict[str, Any],
) -> list[np.ndarray]:
    """
    load image in a thread worker
    """

    res = method(**fn_kwargs)

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
    x_min: Optional[str] = "",
    x_max: Optional[str] = "",
    y_min: Optional[str] = "",
    y_max: Optional[str] = "",
):
    """This function represents the load widget and is called by the wizard to create the widget."""

    # get the default values for the configs
    abs_config_dir = resource_filename("napari_sparrow", "configs")

    with initialize_config_dir(version_base=None, config_dir=abs_config_dir):
        cfg = compose(config_name="pipeline")

    cfg.paths.output_dir = str(output_dir)

    crd = [x_min, x_max, y_min, y_max]
    crd = [0 if val == "" else int(val) for val in crd]

    # TODO fix this. Should Replace None values with 0/size of image in create_sdata function
    if sum( crd )==0:
        crd=None

    cfg.dataset.crop_param = crd
    cfg.dataset.image = path_image

    pipeline = SparrowPipeline(cfg)

    fn_kwargs: Dict[str, Any] = {"pipeline": pipeline}

    worker = _load_worker(method=loadImage, fn_kwargs=fn_kwargs)

    def add_image(sdata: SpatialData, pipeline: SparrowPipeline, layer_name: str):
        """Add the image to the napari viewer, overwrite if it already exists."""

        try:
            # if the layer exists, update its data
            layer = viewer.layers[layer_name]
            viewer.layers.remove(layer)

            log.info(f"Refreshing { layer_name }")
        except KeyError:
            # otherwise add it to the viewer
            log.info(f"Adding { layer_name }")

        offset_x, offset_y = _get_translation(sdata[pipeline.loaded_image_name])

        # Translate image to appear on selected region
        viewer.add_image(
            sdata[pipeline.loaded_image_name].data.squeeze(),
            translate=[
                offset_y,
                offset_x,
            ],
            name=layer_name,
        )

        viewer.layers[layer_name].metadata["pipeline"] = pipeline

        log.info(f"Added { layer_name } layer")

        show_info("Loading finished")

    worker.returned.connect(lambda data: add_image(data, pipeline, utils.LOAD))
    show_info("Loading started")
    worker.start()

    # return worker, so status can be checked for unit testing
    return worker
