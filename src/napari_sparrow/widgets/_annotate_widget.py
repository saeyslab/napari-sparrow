"""
Annotation widget for scoring the genes, returns markergenes and adata objects.
"""
import pathlib
from typing import Any, Callable, Dict, List

import os

import napari
import napari.layers
import napari.types
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from omegaconf.dictconfig import DictConfig
from spatialdata import SpatialData

import napari_sparrow.utils as utils
from napari_sparrow.pipeline import annotate, visualize

log = utils.get_pylogger(__name__)


def annotateImage(
    sdata: SpatialData,
    cfg: DictConfig,
) -> SpatialData:
    """Function representing the annotation step, this calls all the needed functions to annotate the cells with the celltype."""

    sdata, mg_dict = annotate(cfg, sdata)

    sdata = visualize(
        cfg=cfg,
        sdata=sdata,
        mg_dict=mg_dict,
    )

    return sdata


@thread_worker(progress=True)
def _annotation_worker(
    sdata: SpatialData, method: Callable, fn_kwargs: Dict[str, Any]
) -> SpatialData:
    """
    annotate data with marker genes in a thread worker
    """
    return method(sdata, **fn_kwargs)


@magic_factory(
    call_button="Annotate",
    markers_file={"widget_type": "FileEdit", "filter": "*.csv"},
)
def annotate_widget(
    viewer: napari.Viewer,
    markers_file: pathlib.Path = pathlib.Path(""),
    delimiter: str = ",",
    row_norm: bool = False,
    del_celltypes: List[str] = [],
):
    """This function represents the annotation widget and is called by the wizard to create the widget."""

    # Check if a file was passed
    if str(markers_file) in ["", "."]:
        raise ValueError("Please select marker file (.csv)")
    log.info(f"Marker file is {str(markers_file)}")

    # Load data from previous layers
    try:
        allocation_layer=viewer.layers[utils.ALLOCATION]
    except KeyError:
        raise RuntimeError(f"Layer with name '{utils.ALLOCATION}' is not available.")

    try:
        allocation_layer.metadata["adata"]
        sdata=allocation_layer.metadata["sdata"]
        cfg = allocation_layer.metadata["cfg"]
    except KeyError:
        raise RuntimeError(f"Please run allocation step before running annotation step.")

    cfg.dataset.markers = markers_file
    cfg.annotate.row_norm = row_norm
    cfg.annotate.del_celltypes = del_celltypes
    cfg.annotate.delimiter = delimiter

    fn_kwargs = {
        "cfg": cfg,
    }

    worker = _annotation_worker(sdata, annotateImage, fn_kwargs)

    def add_metadata(
        sdata: SpatialData,
        cfg: DictConfig,
        layer_name: str,
    ):
        """Add the metadata to the previous layer, this way it becomes available in the next steps."""

        try:
            # if the layer exists, update its data
            viewer.layers[layer_name]
        except KeyError:
            log.info(f"Layer does not exist {layer_name}")

        # Store data in previous layer

        viewer.layers[layer_name].metadata["adata"] = sdata.table
        viewer.layers[layer_name].metadata["sdata"] = sdata
        viewer.layers[layer_name].metadata["cfg"] = cfg

        utils._export_config( cfg.annotate, os.path.join( cfg.paths.output_dir, 'configs', 'annotate', 'plugin.yaml' ) )

        show_info("Annotation finished")

    worker.returned.connect(lambda data: add_metadata(data, cfg, utils.ALLOCATION))
    show_info("Annotation started")
    worker.start()
