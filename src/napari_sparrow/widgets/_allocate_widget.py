"""
Allocation widget for creating and preprocesing the adata object, filtering the cells and performing clustering.
"""
import pathlib
from typing import Callable

import napari
import napari.layers
import napari.types
import numpy as np
import squidpy.im as sq
from anndata import AnnData
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info

import napari_sparrow.utils as utils
from napari_sparrow import functions as fc

log = utils.get_pylogger(__name__)


def allocateImage(
    path: str,
    ic: sq.ImageContainer,
    masks: np.ndarray,
    pcs: int,
    neighbors: int,
    library_id: str = "melanoma",
    min_size: int = 100,
    max_size: int = 100000,
    cluster_resolution: float = 0.8,
    n_comps: int = 50,
) -> AnnData:
    """Function representing the allocation step, this calls all the needed functions to allocate the transcripts to the cells."""

    adata = fc.create_adata_quick(path, ic, masks, library_id)
    adata, _ = fc.preprocessAdata(adata, masks, n_comps=n_comps)
    adata, _ = fc.filter_on_size(adata, min_size, max_size)
    adata = fc.extract(ic, adata)
    adata = fc.clustering(adata, pcs, neighbors, cluster_resolution)
    return adata


@thread_worker(progress=True)
def _allocation_worker(
    method: Callable,
    fn_kwargs,
) -> AnnData:
    """
    allocate transcripts in a thread worker
    """

    return method(**fn_kwargs)


@magic_factory(
    call_button="Allocate",
    transcripts_file={"widget_type": "FileEdit", "filter": "*.txt"},
)
def allocate_widget(
    viewer: napari.Viewer,
    transcripts_file: pathlib.Path = pathlib.Path(""),
    library_id: str = "melanoma",
    min_size=500,
    max_size=100000,
    pcs: int = 17,
    neighbors: int = 35,
    cluster_resolution: float = 0.8,
    n_components: int = 50,
):
    """This function represents the allocate widget and is called by the wizard to create the widget."""

    # Check if a file was passed
    if str(transcripts_file) in ["", "."]:
        raise ValueError("Please select transcripts file (.txt)")
    log.info(f"Transcripts file is {str(transcripts_file)}")

    # Load data from previous layers
    try:
        ic = viewer.layers[utils.SEGMENT].metadata["ic"]
        masks = viewer.layers[utils.SEGMENT].data_raw
    except KeyError:
        raise RuntimeError("Please run previous steps first")

    fn_kwargs = {
        "path": str(transcripts_file),
        "ic": ic,
        "masks": masks,
        "pcs": pcs,
        "neighbors": neighbors,
        "library_id": library_id,
        "min_size": min_size,
        "max_size": max_size,
        "cluster_resolution": cluster_resolution,
        "n_comps": n_components,
    }

    worker = _allocation_worker(allocateImage, fn_kwargs)

    def add_metadata(result: AnnData):
        """Add the metadata to the previous layer, this way it becomes available in the next steps."""

        try:
            # check if the previous layer exists
            layer = viewer.layers[utils.SEGMENT]
        except KeyError:
            log.info(f"Layer does not exist {utils.SEGMENT}")

        # Store data in previous layer
        layer.metadata["adata"] = result
        layer.metadata["library_id"] = library_id
        layer.metadata["labels_key"] = "cell_ID"
        layer.metadata["points"] = result.uns["spatial"][library_id]["points"]
        layer.metadata["point_diameter"] = 10
        show_info("Allocation finished")

        # Options for napari-spatialData plugin
        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = "um"

    worker.returned.connect(add_metadata)
    show_info("Allocation started")
    worker.start()