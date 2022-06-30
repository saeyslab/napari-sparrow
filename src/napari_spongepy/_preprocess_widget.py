"""
Napari widget for preprocessing raw (Resolve) spatial transcriptomics
microscopy images with nuclear stains. The goal of preprocessing
is to improve the image quality so that subsequent image segmentation
will be more accurate.
"""

from magicgui import magic_factory
import napari
import napari.layers
import napari.types
import napari.utils
from scipy import ndimage
import numpy as np
import cv2
from typing import List


@magic_factory(call_button='Preprocess')
def preprocess_widget(image: napari.layers.Image,
                      tophat_size: int=45,
                      contrast_clip: float=2.5) -> List[napari.types.LayerDataTuple]:
    print(f'About to preprocess {image}; tophat_size={tophat_size} contrast_clip={contrast_clip}')
    if image is None:
        return []

    img = image.data

    # Initialize Napari progress bar.
    prg = napari.utils.progress(total=3)

    # Find mask for inpainting the black tile boundary lines in raw Resolve images. 
    maskLines = np.where(img == 0)  # find the location of the lines 
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask[maskLines[0], maskLines[1]] = 1  # put one values in the correct position
    
    # Perform Navier-Stokes inpainting on the black tile boundary lines.
    prg.set_description('Inpainting tile boundaries')
    inpainted_img = cv2.inpaint(img, inpaintMask=mask, inpaintRadius=15, flags=cv2.INPAINT_NS)
    img = inpainted_img
    prg.update(1)
    
    # Remove background using a tophat filter.
    prg.set_description('Tophat filtering')
    local_minimum_img = ndimage.minimum_filter(img, tophat_size)  
    tophat_filtered_img = img - local_minimum_img
    img = tophat_filtered_img
    prg.update(1)
    
    # Enhance contrast
    prg.set_description('Enhancing contrast')
    clahe = cv2.createCLAHE(clipLimit=contrast_clip, tileGridSize=(8,8))
    img = clahe.apply(img)
    prg.update(1)

    prg.close()

    return [(mask, {'name': 'Missing pixels'}, ),
            (inpainted_img, {'name': 'Inpainted'}, ),
            (tophat_filtered_img, {'name': 'Tophat filtered'}, ),
            (img, {'name': 'Preprocessed'}, )
           ]

