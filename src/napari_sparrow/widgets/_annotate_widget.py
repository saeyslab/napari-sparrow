"""
Annotation widget for scoring the genes, returns markergenes and adata objects.
"""
import os
import pathlib
from typing import Any, Callable, Dict, List, Tuple

import napari
import napari.layers
import napari.types
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from omegaconf.dictconfig import DictConfig
from spatialdata import SpatialData

import napari_sparrow.utils as utils
from napari_sparrow.pipeline_functions import annotate

log = utils.get_pylogger(__name__)


def annotateImage(
    sdata: SpatialData,
    cfg: DictConfig,
) -> Tuple[SpatialData, Dict[str, List[str]]]:
    """Function representing the annotation step, this calls all the needed functions to annotate the cells with the celltype."""

    sdata, mg_dict = annotate(cfg, sdata)

    return sdata, mg_dict


@thread_worker(progress=True)
def _annotation_worker(
    sdata: SpatialData, method: Callable, fn_kwargs: Dict[str, Any]
) -> Tuple[SpatialData, Dict[str, List[str]]]:
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
        sdata = viewer.layers[utils.SEGMENT].metadata["sdata"]
        cfg = viewer.layers[utils.SEGMENT].metadata["cfg"]
    except KeyError:
        raise RuntimeError("Please run previous steps first")

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
        mg_dict: Dict[str, List[str]],
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
        viewer.layers[layer_name].metadata["mg_dict"] = mg_dict

        show_info("Annotation finished")

    worker.returned.connect(lambda data: add_metadata(*data, cfg, utils.SEGMENT))
    show_info("Annotation started")
    worker.start()
