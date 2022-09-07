"""
Napari widget for managing the other widgets and giving a general overview of the workflow.
"""
import pathlib

import napari
import napari.layers
import napari.types
from magicgui import magic_factory

import napari_spongepy.utils as utils
from napari_spongepy import functions as fc

log = utils.get_pylogger(__name__)

current_widget = None


@magic_factory(
    call_button="Visualize", save_folder={"widget_type": "FileEdit", "mode": "d"}
)
def visualize_widget(
    viewer: napari.Viewer,
    save_folder: pathlib.Path = pathlib.Path(""),
):

    adata = viewer.layers[utils.SEGMENT].metadata["adata_annotate"]
    mg_dict = viewer.layers[utils.SEGMENT].metadata["mg_dict"]

    adata, _ = fc.clustercleanliness(adata, list(mg_dict.keys()))

    adata = fc.enrichment(adata)
    fc.save_data(
        adata, str(save_folder) + "/polygons.geojson", str(save_folder) + "/adata.h5ad"
    )

    log.info(f"Data is saved in {str(save_folder)}")
