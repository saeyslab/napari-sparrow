from typing import Optional, Tuple
from spatialdata import SpatialData
import spatialdata
import squidpy as sq
import torch
from spatialdata.transformations import set_transformation, Translation
from cellpose import models
from shapely.affinity import translate

from napari_sparrow.shape._shape import _mask_image_to_polygons
from napari_sparrow.image._image import _substract_translation_crd, _get_translation 


def segmentation_cellpose(
    sdata: SpatialData,
    layer: Optional[str] = None,
    output_layer: str = "segmentation_mask",
    crd: Optional[Tuple[int, int, int, int]] = None,
    device: str = "cpu",
    min_size: int = 80,
    flow_threshold: float = 0.6,
    diameter: int = 55,
    cellprob_threshold: int = 0,
    model_type: str = "nuclei",
    channels=[0, 0],
    chunks="auto",
    lazy=False,
) -> SpatialData:
    """
    Segment images using the Cellpose algorithm and add segmentation results to the SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object.
    layer : Optional[str], default=None
        The image layer in `sdata` to be segmented. If not provided, the last image layer in `sdata` is used.
    output_layer : str, default="segmentation_mask"
        The name of the label layer in which segmentation results will be stored in `sdata`.
    crd : Optional[Tuple[int, int, int, int]], default=None
        The coordinates specifying the region of the image to be segmented. It defines the bounds (x_min, x_max, y_min, y_max).
    device : str, default="cpu"
        Device to run Cellpose on, either "cpu" or "cuda" for GPU.
    min_size : int, default=80
        Minimum size of detected objects.
    flow_threshold : float, default=0.6
        Cellpose flow threshold.
    diameter : int, default=55
        Approximate diameter of objects to be detected.
    cellprob_threshold : int, default=0
        Cellpose cell probability threshold.
    model_type : str, default="nuclei"
        Type of model to be used in Cellpose, options include "nuclei" or "cyto".
    channels : list, default=[0, 0]
        Channels to use in Cellpose. 
        For single channel images, the default value ([0,0]) should not be adapted. 
        For multi channel images, the first element of the list is the channel to segment (count from 1), 
        and the second element is the optional nuclear channel. 
        E.g. for an image with PolyT in second channel, and DAPI in first channel use [2,1] if you want to segment PolyT + nuclei on DAPI; 
        [2,0] if you only want to use PolyT and [1,0] if you only want to use DAPI."
    chunks : str, default="auto"
        The size of the chunks used by cellpose.
    lazy : bool, default=False
        If True, compute segmentation lazily.

    Returns
    -------
    SpatialData
        Updated SpatialData object containing the segmentation mask and boundaries obtained from Cellpose. 
        Segmentation masks will be added as a labels layer to the SpatialData object with name output_layer, 
        and masks boundaries as a shapes layer with name f'{output_layer}_boundaries.'

    Raises
    ------
    ValueError
        If the chosen output_layer name contains the word 'filtered'.

    Notes
    -----
    The function integrates Cellpose segmentation into the SpatialData framework. It manages the pre and post-processing 
    of data, translation of coordinates, and addition of segmentation results back to the SpatialData object.
    """
    
    if "filtered" in output_layer:
        raise ValueError( "Please choose an output_layer name that does not have 'filtered' in its name, "
                        " as these are reserved for filtered out masks and shapes." )

    if layer is None:
        layer = [*sdata.images][-1]

    ic = sq.im.ImageContainer(sdata[layer], layer=layer)

    # crd is specified on original uncropped pixel coordinates
    # need to substract possible translation, because we use crd to crop imagecontainer, which does not take
    # translation into account
    if crd:
        crd = _substract_translation_crd(sdata[layer], crd)
    if crd:
        x0 = crd[0]
        x_size = crd[1] - crd[0]
        y0 = crd[2]
        y_size = crd[3] - crd[2]
        ic = ic.crop_corner(y=y0, x=x0, size=(y_size, x_size))

        # rechunk if you take crop, in order to be able to save as spatialdata object.
        # TODO check if this still necessary
        # for layer in ic.data.data_vars:
        #    chunksize = ic[layer].data.chunksize[0]
        #    ic[layer] = ic[layer].chunk(chunksize)

    tx, ty = _get_translation(sdata[layer])

    sq.im.segment(
        img=ic,
        layer=layer,
        method=_cellpose,
        channel=None,
        chunks=chunks,
        lazy=lazy,
        min_size=min_size,
        layer_added=output_layer,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        diameter=diameter,
        model_type=model_type,
        channels=channels,
        device=device,
    )

    if crd:
        tx = tx + crd[0]
        ty = ty + crd[2]

    translation = Translation([tx, ty], axes=("x", "y"))

    temp = ic.data.rename_dims({"channels": "c"})
    spatial_label = spatialdata.models.Labels2DModel.parse(
        temp[output_layer].squeeze().transpose("y", "x")
    )

    set_transformation(spatial_label, translation)

    # during adding of image it is written to zarr store
    sdata.add_labels(name=output_layer, labels=spatial_label)

    polygons = _mask_image_to_polygons(mask=sdata[output_layer].data)
    polygons = polygons.dissolve(by="cells")

    x_translation, y_translation = _get_translation(sdata[output_layer])
    polygons["geometry"] = polygons["geometry"].apply(
        lambda geom: translate(geom, xoff=x_translation, yoff=y_translation)
    )

    shapes_name = f"{output_layer}_boundaries"

    sdata.add_shapes(
        name=shapes_name,
        shapes=spatialdata.models.ShapesModel.parse(polygons),
    )

    return sdata


def _cellpose(
    img,
    min_size=80,
    cellprob_threshold=-4,
    flow_threshold=0.85,
    diameter=100,
    model_type="cyto",
    channels=[1, 0],
    device="cpu",
):
    gpu = torch.cuda.is_available()
    model = models.Cellpose(gpu=gpu, model_type=model_type, device=torch.device(device))
    masks, _, _, _ = model.eval(
        img,
        diameter=diameter,
        channels=channels,
        min_size=min_size,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    return masks
