"""
Annotation widget for scoring the genes, returns markergenes and adata objects.
"""
import pathlib
from typing import Callable

import napari
import napari.layers
import napari.types
from anndata import AnnData
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from sparrow import functions as fc

import napari_sparrow.utils as utils

log = utils.get_pylogger(__name__)


def annotateImage(
    adata: AnnData, path_marker_genes: str, row_norm: bool = False
) -> dict:
    """Function representing the annotation step, this calls all the needed functions to annotate the cells with the celltype."""

    mg_dict, _ = fc.scoreGenes(adata, path_marker_genes, row_norm)
    return mg_dict


@thread_worker(progress=True)
def _annotation_worker(method: Callable, fn_kwargs) -> dict:
    """
    annotate data with marker genes in a thread worker
    """
    return method(**fn_kwargs)


@magic_factory(
    call_button="Annotate",
    markers_file={"widget_type": "FileEdit", "filter": "*.csv"},
)
def annotate_widget(
    viewer: napari.Viewer,
    markers_file: pathlib.Path = pathlib.Path(""),
    row_norm: bool = False,
):
    """This function represents the annotation widget and is called by the wizard to create the widget."""

    # Check if a file was passed
    if str(markers_file) in ["", "."]:
        raise ValueError("Please select marker file (.csv)")
    log.info(f"Marker file is {str(markers_file)}")

    # Load data from previous layers
    try:
        adata = viewer.layers[utils.SEGMENT].metadata["adata"]
    except KeyError:
        raise RuntimeError("Please run previous steps first")

    fn_kwargs = {
        "adata": adata,
        "path_marker_genes": str(markers_file),
        "row_norm": row_norm,
    }

    worker = _annotation_worker(annotateImage, fn_kwargs)

    def add_metadata(result: dict):
        """Add the metadata to the previous layer, this way it becomes available in the next steps."""

        try:
            # check if the previous layer exists
            layer = viewer.layers[utils.SEGMENT]
        except KeyError:
            log.info(f"Layer does not exist {utils.SEGMENT}")

        # Store data in previous layer
        layer.metadata["mg_dict"] = result
        show_info("Annotation finished")

    worker.returned.connect(add_metadata)
    show_info("Annotation started")
    worker.start()
