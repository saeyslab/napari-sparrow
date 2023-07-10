"""
Visualisation widget for saving the geojson and adata objects.
"""
import os
import pathlib
from typing import Callable, Dict, List

import napari
import napari.layers
import napari.types
from anndata import AnnData
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from omegaconf.dictconfig import DictConfig
from spatialdata import SpatialData

import napari_sparrow.utils as utils
from napari_sparrow.pipeline_functions import visualize

log = utils.get_pylogger(__name__)


def exportImage(
    sdata: SpatialData,
    mg_dict: Dict[str, List[str]],
    cfg: DictConfig,
):
    """Function representing the visualization step, this calls all the needed functions to save the data to a directory."""

    sdata = visualize(
        cfg=cfg,
        sdata=sdata,
        mg_dict=mg_dict,
    )


    return sdata


@thread_worker(progress=True)
def _exporting_worker(
    sdata: SpatialData,
    method: Callable,
    fn_kwargs,
):
    """
    save data in a thread worker
    """
    return method(sdata, **fn_kwargs)


@magic_factory(
    call_button="Visualize",
)
def export_widget(
    viewer: napari.Viewer,
):
    """This function represents the visualize widget and is called by the wizard to create the widget."""

    # Load data from previous layer
    try:
        sdata = viewer.layers[utils.SEGMENT].metadata["sdata"]
        cfg = viewer.layers[utils.SEGMENT].metadata["cfg"]
        mg_dict = viewer.layers[utils.SEGMENT].metadata["mg_dict"]

    except KeyError:
        raise RuntimeError("Please run previous steps first")

    fn_kwargs = {
        "cfg": cfg,
        "mg_dict": mg_dict,
    }

    worker = _exporting_worker(sdata, exportImage, fn_kwargs)

    def add_metadata(
        sdata: SpatialData,
        cfg: DictConfig,
        layer_name: str,
    ):
        """Add the metadata to the previous layer, this way it becomes available in the next steps."""

        print("add metadata", sdata)

        try:
            # if the layer exists, update its data
            viewer.layers[layer_name]
        except KeyError:
            log.info(f"Layer does not exist {layer_name}")

        # Store data in previous layer

        viewer.layers[layer_name].metadata["adata"] = sdata.table
        viewer.layers[layer_name].metadata["sdata"] = sdata
        viewer.layers[layer_name].metadata["cfg"] = cfg

        #sdata.write(os.path.join(cfg.paths.output_dir, "sdata_export.zarr"))

        show_info("Exporting finished")

    worker.returned.connect(lambda data: add_metadata(data, cfg, utils.SEGMENT))
    show_info("Exporting started")
    worker.start()
