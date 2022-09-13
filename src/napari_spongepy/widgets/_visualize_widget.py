"""
Visualisation widget for saving the geojson and adata objects.
"""
import pathlib
from typing import Callable, List

import napari
import napari.layers
import napari.types
from anndata import AnnData
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info

import napari_spongepy.utils as utils
from napari_spongepy import functions as fc

log = utils.get_pylogger(__name__)


def visualizeImage(adata: AnnData, genes: List[str], save_folder: str):
    adata, _ = fc.clustercleanliness(adata, genes)
    adata = fc.enrichment(adata)
    fc.save_data(adata, save_folder + "/polygons.geojson", save_folder + "/adata.h5ad")


@thread_worker(progress=True)
def _visualisation_worker(
    method: Callable,
    fn_kwargs,
):
    """
    save data in a thread worker
    """
    method(**fn_kwargs)


@magic_factory(
    call_button="Visualize",
    save_folder={"widget_type": "FileEdit", "mode": "d"},
)
def visualize_widget(
    viewer: napari.Viewer,
    save_folder: pathlib.Path = pathlib.Path(""),
):
    # Check if a directory was passed
    if str(save_folder) in ["", "."]:
        raise ValueError("Please select output folder")
    log.info(f"Data will be saved in {str(save_folder)}")

    # Load data from previous layer
    try:
        adata = viewer.layers[utils.SEGMENT].metadata["adata"]
        mg_dict = viewer.layers[utils.SEGMENT].metadata["mg_dict"]
    except KeyError:
        raise RuntimeError("Please run previous steps first")

    fn_kwargs = {
        "adata": adata,
        "genes": list(mg_dict.keys()),
        "save_folder": str(save_folder),
    }

    worker = _visualisation_worker(visualizeImage, fn_kwargs)

    worker.returned.connect(
        lambda: show_info(f"Visualisation finished, files saved in: {str(save_folder)}")
    )
    show_info("Visualisation started")
    worker.start()
