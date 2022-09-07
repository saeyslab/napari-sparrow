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
    call_button="Annotate", markers_file={"widget_type": "FileEdit", "filter": "*.csv"}
)
def annotate_widget(
    viewer: napari.Viewer,
    markers_file: pathlib.Path = pathlib.Path(""),
    row_norm: bool = False,
):

    adata = viewer.layers[utils.SEGMENT].metadata["adata_allocate"]
    log.info(f"Marker file is {markers_file}")

    mg_dict, _ = fc.scoreGenes(adata, str(markers_file), row_norm)

    viewer.layers[utils.SEGMENT].metadata["mg_dict"] = mg_dict
    viewer.layers[utils.SEGMENT].metadata["adata_annotate"] = adata
    log.info(f"mg_dict is {mg_dict}")
