"""
Napari widget for cell segmentation of 
preprocessed (Resolve) spatial transcriptomics
microscopy images with nuclear stains.
Segmentation is performed with Cellpose.
"""

from magicgui import magic_factory
import napari
import napari.layers
import napari.types
import numpy as np
from typing import List
from napari.qt.threading import thread_worker


@thread_worker
def _segmentation_worker(img: np.ndarray,
                         use_gpu: bool,
                         min_size: int,
                         flow_threshold: float,
                         diameter: int,
                         mask_threshold: int) -> np.ndarray:

    # We delay importing the large cellpose module to when it is really needed over here.
    # This makes the segmentation widget appear almost instantaneously when it is opened in napari,
    # otherwise it would only appear after the cellpose package is imported, which takes several seconds.
    from cellpose import models

    model = models.Cellpose(gpu=use_gpu, model_type='nuclei')
    channels = np.array([0, 0])

    masks, _, _, _ = model.eval(img,
                                diameter=diameter,
                                channels=channels,
                                min_size=min_size,
                                flow_threshold=flow_threshold,
                                cellprob_threshold=mask_threshold)
    return masks


@magic_factory(call_button='Segment')
def segmentation_widget(viewer: napari.Viewer,
                        image: napari.layers.Image,
                        use_gpu: bool=True,
                        min_size: int=80,
                        flow_threshold: float=0.6,
                        diameter: int=55,
                        mask_threshold: int=0):

    print(f'About to segment {image} using Cellpose; use_gpu={use_gpu}')
    if image is None:
        return

    worker = _segmentation_worker(image.data, use_gpu, min_size, flow_threshold, diameter, mask_threshold)
    worker.returned.connect(lambda label_img: viewer.add_labels(label_img, name='Segmentation'))
    worker.start()
