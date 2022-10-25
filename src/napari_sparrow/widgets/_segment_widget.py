"""
Napari widget for cell segmentation of
cleaned (Resolve) spatial transcriptomics
microscopy images with nuclear stains.
Segmentation is performed with Squidpy ImageContainer and segment.
"""

from enum import Enum
from typing import Callable, List, Tuple

import napari
import napari.layers
import napari.types
import numpy as np
import squidpy.im as sq
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info

import napari_sparrow.utils as utils

log = utils.get_pylogger(__name__)


class ModelOption(Enum):
    nuclei = "nuclei"
    cyto = "cyto"


def segmentImage(
    ic: sq.ImageContainer,
    device: str = "cpu",
    min_size: int = 80,
    flow_threshold: float = 0.6,
    diameter: int = 55,
    cellprob_threshold: int = 0,
    model_type: str = "nuclei",
    channels: List[int] = [0, 0],
    left_corner: Tuple[int, int] = None,
    size: Tuple[int, int] = None,
) -> Tuple[np.ndarray, sq.ImageContainer]:
    """Function representing the segmentation step, this calls the segmentation function."""
    from sparrow.functions import segmentation

    # Crop imageContainer
    if left_corner is not None and size is not None:
        ic = ic.crop_corner(*left_corner, size)

    mask, _, _, ic = segmentation(
        ic,
        device,
        min_size,
        flow_threshold,
        diameter,
        cellprob_threshold,
        model_type,
        channels,
    )

    return mask, ic


@thread_worker(progress=True)
def _segmentation_worker(
    img: np.ndarray,
    method: Callable,
    fn_kwargs=None,
) -> Tuple[np.ndarray, sq.ImageContainer]:
    """
    segment image in a thread worker
    """

    return method(img, **fn_kwargs)


@magic_factory(
    call_button="Segment",
    cell_threshold={"widget_type": "SpinBox", "min": -50, "max": 100},
    channels={"layout": "vertical", "options": {"min": 0, "max": 3}},
)
def segment_widget(
    viewer: napari.Viewer,
    image: napari.layers.Image,
    subset: napari.layers.Shapes,
    device: str = "cpu",
    min_size: int = 80,
    flow_threshold: float = 0.8,
    diameter: int = 50,
    cell_threshold: int = -2,
    model_type: ModelOption = ModelOption.nuclei,
    channels: List[int] = [0, 0],
):
    """This function represents the segment widget and is called by the wizard to create the widget."""

    if image is None:
        raise ValueError("Please select an image")

    fn_kwargs = {
        "device": device,
        "min_size": min_size,
        "flow_threshold": flow_threshold,
        "diameter": diameter,
        "cellprob_threshold": cell_threshold,
        "model_type": model_type.value,
        "channels": channels,
    }

    # Subset shape
    if subset:
        # Check if shapes layer only holds one shape and shape is rectangle
        if len(subset.shape_type) != 1 or subset.shape_type[0] != "rectangle":
            raise ValueError("Please select one rectangular subset")

        coordinates = np.array(subset.data[0])
        left_corner = coordinates[coordinates.sum(axis=1).argmin()].astype(int)
        size = (
            int(coordinates[:, 0].max() - coordinates[:, 0].min()),
            int(coordinates[:, 1].max() - coordinates[:, 1].min()),
        )

        fn_kwargs["left_corner"] = left_corner
        fn_kwargs["size"] = size

    # Load imageContainer from previous layer
    if image.name == utils.CLEAN:
        ic = viewer.layers[image.name].metadata["ic"]

        # If we select the cleaned image which is cropped, adjust for corner coordinates offset
        if subset:
            fn_kwargs["left_corner"] = left_corner - np.array(
                [ic.data.attrs["coords"].y0, ic.data.attrs["coords"].x0]
            )
    # Create new imageContainer
    else:
        ic = sq.ImageContainer(image.data_raw)

    worker = _segmentation_worker(ic, segmentImage, fn_kwargs=fn_kwargs)

    def add_label(mask: np.ndarray, ic: sq.ImageContainer, layer_name: str):
        """Add the label to the napari viewer, overwrite if it already exists."""
        try:
            # if the layer exists, update its data
            layer = viewer.layers[layer_name]
            viewer.layers.remove(layer)

            log.info(f"Refreshing {layer_name}")
        except KeyError:
            # otherwise add it to the viewer
            log.info(f"Adding {layer_name}")

        # Translate image to appear on selected region
        viewer.add_labels(
            mask,
            visible=True,
            name=layer_name,
            translate=[ic.data.attrs["coords"].y0, ic.data.attrs["coords"].x0],
        )

        viewer.layers[utils.SEGMENT].metadata["ic"] = ic
        show_info("Segmentation finished")

    worker.returned.connect(lambda data: add_label(*data, utils.SEGMENT))  # type: ignore
    show_info(
        "Segmentation started" + ", CPU selected: might take some time"
        if device == "cpu"
        else ""
    )
    worker.start()
