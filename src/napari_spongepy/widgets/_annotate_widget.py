"""
Annotation widget for scoring the genes, returns markergenes and adata objects.
"""
import pathlib
from typing import Callable, Tuple

import napari
import napari.layers
import napari.types
from anndata import AnnData
from magicgui import magic_factory
from napari.qt.threading import thread_worker

import napari_spongepy.utils as utils
from napari_spongepy import functions as fc

log = utils.get_pylogger(__name__)

current_widget = None


def annotateImage(
    adata: AnnData, path_marker_genes: str, row_norm: bool = False
) -> Tuple[dict, AnnData]:
    return fc.scoreGenes(adata, path_marker_genes, row_norm)


@thread_worker(progress=True)
def _annotation_worker(method: Callable, fn_kwargs) -> Tuple[dict, AnnData]:
    res = method(**fn_kwargs)

    return res


@magic_factory(
    call_button="Annotate",
    markers_file={"widget_type": "FileEdit", "filter": "*.csv"},
    result_widget=True,
)
def annotate_widget(
    viewer: napari.Viewer,
    markers_file: pathlib.Path = pathlib.Path(""),
    row_norm: bool = False,
) -> str:

    if str(markers_file) in ["", "."]:
        return "Please select marker file (.csv)"
    log.info(f"Marker file is {markers_file}")

    try:
        adata = viewer.layers[utils.SEGMENT].metadata["adata_allocate"]
    except KeyError:
        return "Please run previous steps first"

    fn_kwargs = {
        "adata": adata,
        "path_marker_genes": str(markers_file),
        "row_norm": row_norm,
    }

    worker = _annotation_worker(annotateImage, fn_kwargs)
    log.info("Annotation worker created")

    def add_metadata(result: Tuple[dict, AnnData]):
        try:
            # if the layer exists, update its data
            layer = viewer.layers[utils.SEGMENT]
        except KeyError:
            # otherwise add it to the viewer
            log.info(f"Layer does not exist {utils.SEGMENT}")

        layer.metadata["mg_dict"] = result[0]
        layer.metadata["adata_annotate"] = result[1]
        log.info("Annotation finished")

        return "Annotation finished"

    worker.returned.connect(add_metadata)
    worker.start()
    return "Annotation started"
