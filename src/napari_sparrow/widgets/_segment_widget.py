"""
Napari widget for cell segmentation of
cleaned (Resolve) spatial transcriptomics
microscopy images with nuclear stains.
Segmentation is performed with Squidpy ImageContainer and segment.
"""

import os
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import napari
import napari.layers
import napari.types
import numpy as np
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from omegaconf.dictconfig import DictConfig
from spatialdata import SpatialData
from spatialdata.transformations import Translation, set_transformation

import napari_sparrow.utils as utils
from napari_sparrow.image._image import _get_translation
from napari_sparrow.io import create_sdata
from napari_sparrow.pipeline import segment

log = utils.get_pylogger(__name__)


class ModelOption(Enum):
    nuclei = "nuclei"
    cyto = "cyto"


def segmentImage(
    sdata: SpatialData,
    cfg: DictConfig,
) -> Tuple[SpatialData, DictConfig]:
    """Function representing the segmentation step, this calls the segmentation function."""

    sdata = segment(cfg, sdata)

    return sdata


@thread_worker(progress=True)
def _segmentation_worker(
    sdata: SpatialData,
    method: Callable,
    fn_kwargs: Dict[str, Any],
) -> SpatialData:
    """
    segment image in a thread worker
    """

    return method(sdata, **fn_kwargs)


@magic_factory(
    call_button="Segment",
    cellprob_threshold={"widget_type": "SpinBox", "min": -50, "max": 100},
    channels={"layout": "vertical", "options": {"min": 0, "max": 3}},
)
def segment_widget(
    viewer: napari.Viewer,
    image: Optional[napari.layers.Image] = None,
    subset: Optional[napari.layers.Shapes] = None,
    device: str = "cpu",
    min_size: int = 80,
    flow_threshold: float = 0.8,
    diameter: int = 50,
    cellprob_threshold: int = -2,
    model_type: ModelOption = ModelOption.nuclei,
    channels: List[int] = [1, 0],
    voronoi_radius: int = 0,
    chunks: int = 2048,
    depth: int = 100,
):
    """This function represents the segment widget and is called by the wizard to create the widget."""

    if image is None:
        raise ValueError("Please select an image")

    fn_kwargs: Dict[str, Any] = {}

    # Load SpatialData object from previous layer
    if image.name == utils.CLEAN:
        cfg = viewer.layers[utils.CLEAN].metadata["cfg"]
        sdata = viewer.layers[image.name].metadata["sdata"]

        for shapes_layer in [*sdata.shapes]:
            del sdata.shapes[shapes_layer]

        for labels in [*sdata.labels]:
            del sdata.labels[labels]

    # Create new spatialdata
    elif image.name == utils.LOAD:
        # update this
        if len(image.data_raw.shape) == 3:
            dims = ["c", "y", "x"]
        elif len(image.data_raw.shape) == 2:
            dims = ["y", "x"]

        # We need to create new sdata object, because sdata object in
        # viewer.layers[utils.LOAD].metadata["sdata"][utils.LOAD] is backed by .zarr store
        # and we are not allowed to overwrite it (i.e. we would not be allowed to run the cleaning step twice)
        cfg = viewer.layers[utils.LOAD].metadata["cfg"]

        sdata = create_sdata(
            input=image.data_raw, img_layer="raw_image", chunks=1024, dims=dims
        )

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
            f"Please run the cleaning step on the layer with name '{utils.LOAD}' or '{utils.CLEAN}',"
            f"it seems layer with name '{image.name}' was selected."
        )

    # Subset shape
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

        cfg.segmentation.crop_param = crd

    else:
        cfg.segmentation.crop_param = None

    # update config
    cfg.device = device
    cfg.segmentation.small_size_vis = cfg.segmentation.crop_param
    cfg.segmentation.min_size = min_size
    cfg.segmentation.flow_threshold = flow_threshold
    cfg.segmentation.diameter = diameter
    cfg.segmentation.cellprob_threshold = cellprob_threshold
    cfg.segmentation.model_type = model_type.value
    cfg.segmentation.channels = channels
    cfg.segmentation.chunks = chunks
    cfg.segmentation.depth = depth
    cfg.segmentation.voronoi_radius = voronoi_radius
    # we override default settings, because for plugin, we want to keep things in memory,
    # otherwise export step would redo segmentation step.
    #cfg.segmentation.lazy = False

    fn_kwargs["cfg"] = cfg

    worker = _segmentation_worker(sdata, segmentImage, fn_kwargs=fn_kwargs)

    def add_shape(sdata: SpatialData, cfg: DictConfig, layer_name: str):
        """Add the shapes to the napari viewer, overwrite if it already exists."""
        try:
            # if the layer exists, update its data
            layer = viewer.layers[layer_name]
            viewer.layers.remove(layer)

            log.info(f"Refreshing {layer_name}")
        except KeyError:
            # otherwise add it to the viewer
            log.info(f"Adding {layer_name}")

        if cfg.segmentation.voronoi_radius:
            shapes_layer = f"expanded_cells{cfg.segmentation.voronoi_radius}"
        else:
            shapes_layer = cfg.segmentation.output_shapes_layer

        polygons = utils._get_polygons_in_napari_format(df=sdata.shapes[shapes_layer])

        show_info("Adding segmentation shapes, this can be slow on large images...")
        viewer.add_shapes(
            polygons,
            name=layer_name,
            shape_type="polygon",
            edge_color="coral",
            face_color="royalblue",
            edge_width=2,
            opacity=0.5,
        )

        # show_info( "Adding segmentation labels." )
        # Translate image to appear on selected region
        """
        viewer.add_labels(
            sdata.labels[ cfg.segmentation.output_layer ].data.squeeze(),
            visible=True,
            name=layer_name,
            translate=[
                offset_y,
                offset_x,
            ],
        )
        """

        viewer.layers[layer_name].metadata["sdata"] = sdata
        # we need the original shapes, in order for next step (allocation) to be able to run multiple times
        viewer.layers[layer_name].metadata["shapes"] = sdata.shapes.copy()
        viewer.layers[layer_name].metadata["cfg"] = cfg

        log.info(f"Added {utils.SEGMENT} layer")

        utils._export_config(
            cfg.segmentation,
            os.path.join(
                cfg.paths.output_dir, "configs", "segmentation", "plugin.yaml"
            ),
        )

        show_info("Segmentation finished")

    worker.returned.connect(lambda data: add_shape(data, cfg, utils.SEGMENT))  # type: ignore
    show_info(
        "Segmentation started" + ", CPU selected: might take some time"
        if device == "cpu"
        else ""
    )
    worker.start()

    return worker
