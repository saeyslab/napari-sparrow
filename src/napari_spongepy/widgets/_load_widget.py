"""
Napari widget for managing the other widgets and giving a general overview of the workflow.
"""
import pathlib

import napari
import napari.layers
import napari.types
from magicgui import magic_factory
from skimage import io

import napari_spongepy.utils as utils

log = utils.get_pylogger(__name__)


@magic_factory(
    call_button="Load", image_file={"widget_type": "FileEdit", "filter": "*.tiff"}
)
def load_widget(
    viewer: napari.Viewer,
    image_file: pathlib.Path = pathlib.Path(""),
):
    layer_name = utils.IMAGE

    def add_labels(img):
        try:
            # if the layer exists, update its data
            layer = viewer.layers[layer_name]
            viewer.layers.remove(layer)

            log.info(f"Refreshing {layer_name}")
        except KeyError:
            # otherwise add it to the viewer
            log.info(f"Adding {layer_name}")
        viewer.add_labels(img, visible=True, name=layer_name)

        return viewer

    img = io.imread(str(image_file))
    add_labels(img)
    log.info(f"image is {str(image_file)}")
