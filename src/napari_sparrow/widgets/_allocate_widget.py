"""
Allocation widget for creating and preprocesing the adata object, filtering the cells and performing clustering.
"""
import os
import pathlib
from typing import Any, Callable, Dict, Optional


import napari
import napari.layers
import napari.types

from magicgui import magic_factory
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from omegaconf.dictconfig import DictConfig
import spatialdata
from spatialdata import SpatialData

import napari_sparrow.utils as utils
from napari_sparrow.pipeline import allocate

log = utils.get_pylogger(__name__)


def allocateImage(
    sdata: SpatialData,
    cfg: DictConfig,
) -> SpatialData:
    """Function representing the allocation step, this calls all the needed functions to allocate the transcripts to the cells."""

    sdata = allocate(cfg, sdata)

    return sdata


@thread_worker(progress=True)
def _allocation_worker(
    sdata: SpatialData,
    method: Callable,
    fn_kwargs: Dict[str, Any],
) -> SpatialData:
    """
    allocate transcripts in a thread worker
    """

    return method(sdata, **fn_kwargs)


@magic_factory(
    call_button="Allocate",
    transcripts_file={"widget_type": "FileEdit"},
    transform_matrix={"widget_type": "FileEdit"},
)
def allocate_widget(
    viewer: napari.Viewer,
    transcripts_file: pathlib.Path = pathlib.Path(""),
    delimiter: str = "\t",
    header: bool = False,
    column_x: int = 0,
    column_y: int = 1,
    column_gene: int = 3,
    midcount: bool = False,
    column_midcount: Optional[int] = None,
    transform_matrix: Optional[pathlib.Path] = None,
    min_counts:int=10,
    min_cells:int=5,
    size_normalization: bool = True,
    n_comps: int = 50,
    min_size: int = 500,
    max_size: int = 100000,
    pcs: int = 17,
    neighbors: int = 35,
    cluster_resolution: float = 0.8,
):
    """This function represents the allocate widget and is called by the wizard to create the widget."""

    # Check if a file was passed
    if str(transcripts_file) in ["", "."]:
        raise ValueError("Please select transcripts file (.txt)")
    log.info(f"Transcripts file is {str(transcripts_file)}")

    # Load data from previous layers
    try:
        segment_layer = viewer.layers[utils.SEGMENT]
    except:
        raise RuntimeError(f"Layer with name '{utils.SEGMENT}' is not available")

    try:
        sdata = segment_layer.metadata["sdata"]
        shapes = segment_layer.metadata["shapes"]
        cfg = segment_layer.metadata["cfg"]
    except KeyError:
        raise RuntimeError(
            f"Please run segmentation step before running allocation step."
        )

    # need to add original unfiltered shapes to sdata object at the beginning of the allocation step.
    # otherwise polygons that were filtered out would not be available any more if you do a rerun of the allocation step.
    for shapes_name in [*shapes]:
        sdata.add_shapes(
            name=shapes_name,
            shapes=spatialdata.models.ShapesModel.parse(shapes[shapes_name]),
            overwrite=True,
        )

    # napari widget does not support the type Optional[int], therefore only choose whether there is a header or not,
    # and do same for midcount column
    if header:
        header = 0
    else:
        header = None

    if midcount:
        column_midcount = column_midcount
    else:
        column_midcount = None

    cfg.dataset.coords = transcripts_file
    if transform_matrix:
        cfg.dataset.transform_matrix = transform_matrix

    cfg.allocate.size_norm = size_normalization
    cfg.allocate.min_counts= min_counts
    cfg.allocate.min_cells=min_cells
    cfg.allocate.n_comps = n_comps
    cfg.allocate.min_size = min_size
    cfg.allocate.max_size = max_size
    cfg.allocate.pcs = pcs
    cfg.allocate.neighbors = neighbors
    cfg.allocate.cluster_resolution = cluster_resolution
    cfg.allocate.delimiter = delimiter
    cfg.allocate.header = header
    cfg.allocate.column_x = column_x
    cfg.allocate.column_y = column_y
    cfg.allocate.column_gene = column_gene
    cfg.allocate.column_midcount = column_midcount

    fn_kwargs = {
        "cfg": cfg,
    }

    worker = _allocation_worker(sdata, allocateImage, fn_kwargs=fn_kwargs)

    def add_metadata(sdata: SpatialData, cfg: DictConfig, layer_name: str):
        """Update the polygons, add anndata object to the metadata, so it can be viewed via napari spatialdata plugin, and it
        becomes visible in next steps."""

        try:
            # if the layer exists, update its data
            layer = viewer.layers[layer_name]
            viewer.layers.remove(layer)
            log.info(f"Refreshing {layer_name}")

        except KeyError:
            log.info(f"Layer '{layer_name}' does not exist.")

        if cfg.segmentation.voronoi_radius:
            shapes_layer = f"expanded_cells{cfg.segmentation.voronoi_radius}"
        else:
            shapes_layer = cfg.segmentation.output_shapes_layer

        polygons = utils._get_polygons_in_napari_format(df=sdata.shapes[shapes_layer])

        # we add the polygons again in this step, because some of them are filtered out in the allocate step
        # (i.e. due to size of polygons etc). If we do not update the polygons here, napari complains because
        # number of polygons does not match any more with e.g. number of polygons with a leiden cluster assigned.
        show_info(
            "Adding updated segmentation shapes, this can be slow on large images..."
        )
        viewer.add_shapes(
            polygons,
            name=layer_name,
            shape_type="polygon",
            edge_color="coral",
            face_color="royalblue",
            edge_width=2,
            opacity=0.5,
        )

        viewer.layers[layer_name].metadata["adata"] = sdata.table
        viewer.layers[layer_name].metadata["sdata"] = sdata
        viewer.layers[layer_name].metadata["cfg"] = cfg

        log.info( f"Added {utils.ALLOCATION} layer" )

        utils._export_config( cfg.allocate, os.path.join( cfg.paths.output_dir, 'configs', 'allocate', 'plugin.yaml' ) )

        show_info("Allocation finished")

        # Options for napari-spatialData plugin
        viewer.scale_bar.visible = True

    worker.returned.connect(lambda data: add_metadata(data, cfg, f"{utils.ALLOCATION}"))
    show_info("Allocation started")
    worker.start()

    return worker