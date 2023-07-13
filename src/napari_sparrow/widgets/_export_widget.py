"""
Visualisation widget for saving the geojson and adata objects.
"""
import os

import napari
import napari.layers
import napari.types
from magicgui import magic_factory
from napari.utils.notifications import show_info

import napari_sparrow.utils as utils

@magic_factory(
    call_button="Export",
)
def export_widget(
    viewer: napari.Viewer,
):
    """This function represents the export widget and is called by the wizard to create the widget."""

    # Load data from previous layers
    try:
        segment_layer=viewer.layers[utils.SEGMENT]

    except:
        raise RuntimeError(f"Layer with name '{utils.SEGMENT}' is not available")

    try:
        sdata = segment_layer.metadata["sdata"]
        cfg = segment_layer.metadata["cfg"]
    except KeyError:
        raise RuntimeError(f"Please run pipeline at least up to segmentation step before running export step.")

    show_info("Exporting started")

    sdata.write(os.path.join(cfg.paths.output_dir, "sdata_export.zarr"))

    show_info("Exporting finished")