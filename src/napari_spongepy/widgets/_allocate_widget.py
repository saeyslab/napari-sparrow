"""
Allocation widget for creating and preprocesing the adata object, filtering the cells and performing clustering.
"""
import pathlib

import napari
import napari.layers
import napari.types
from magicgui import magic_factory
from napari.qt.threading import thread_worker
import numpy as np
from typing import Callable
from anndata import AnnData
import napari_spongepy.utils as utils
from napari_spongepy import functions as fc

log = utils.get_pylogger(__name__)


def allocateImage(path: str, img: np.ndarray, masks: np.ndarray, pcs: int, neighbors: int, library_id: str = "melanoma", min_size: int = 100, max_size: int = 100000, cluster_resolution: float = 0.8) -> AnnData:
    adata = fc.create_adata_quick(path, img, masks, library_id)
    adata, _ = fc.preprocessAdata(adata, masks)
    adata, _ = fc.filter_on_size(adata, min_size, max_size)
    fc.clustering(adata, pcs, neighbors, cluster_resolution)
    return adata


@thread_worker(progress=True)
def _allocation_worker(
    method: Callable,
    fn_kwargs,
) -> AnnData:
    res = method(**fn_kwargs)

    return res


@magic_factory(
    call_button="Allocate",
    transcripts_file={"widget_type": "FileEdit", "filter": "*.txt"}, result_widget=True
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
):

    if str(transcripts_file) in ["", "."]:
        return "Please select transcripts file (.txt)"
    log.info(f"Transcripts file is {str(transcripts_file)}")

    try:
        img = viewer.layers[utils.CLEAN].data_raw
        masks = viewer.layers[utils.SEGMENT].data_raw
    except KeyError:
        return "Please run previous steps first"

    fn_kwargs = {
        "path": str(transcripts_file),
        "img": img,
        "masks": masks,
        "pcs": pcs,
        "neighbors": neighbors,
        "library_id": library_id,
        "min_size": min_size,
        "max_size": max_size,
        "cluster_resolution": cluster_resolution
    }

    worker = _allocation_worker(allocateImage, fn_kwargs)
    log.info("Allocationn worker Created")

    def add_metadata(result: AnnData):
        try:
            # if the layer exists, update its data
            layer = viewer.layers[utils.SEGMENT]
        except KeyError:
            # otherwise add it to the viewer
            log.info(f"Layer does not exist {utils.SEGMENT}")
        
        layer.metadata["adata_allocate"] = result
        log.info("Allocation finished")
        
        return "Allocation finished" 

    worker.returned.connect(add_metadata)
    worker.start()
