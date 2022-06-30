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
import torch
from cellpose import models


@magic_factory(call_button='Segment')
def segmentation_widget(image: napari.layers.Image,
                        use_gpu: bool=True,
                        min_size: int=80,
                        flow_threshold: float=0.6,
                        diameter: int=55,
                        mask_threshold: int=0) -> List[napari.types.LayerDataTuple]:

    print(f'About to segment {image} using Cellpose; use_gpu={use_gpu}')
    if image is None:
        return []

    img = image.data

    model = models.Cellpose(gpu=use_gpu, model_type='nuclei')
    channels = np.array([0,0])
    
    masks, flows, styles, diams = model.eval(img, 
                                             diameter=diameter,
                                             channels=channels,
                                             min_size=min_size,
                                             flow_threshold=flow_threshold,
                                             cellprob_threshold=mask_threshold)

    return [(masks, {'name': 'Segmentation'}, 'labels')]
