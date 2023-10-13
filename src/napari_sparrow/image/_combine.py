from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union

import dask.array as da
from dask.array import Array
from spatial_image import SpatialImage
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import Translation
from xarray import DataArray

from napari_sparrow.image._image import (
    _add_image_layer,
    _get_spatial_element,
    _get_translation,
    _substract_translation_crd,
)
from napari_sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def combine(
    sdata: SpatialData,
    img_layer: Optional[str] = None,
    output_layer: Optional[str] = None,
    nuc_channels: Optional[int | Iterable[int]] = None,
    mem_channels: Optional[int | Iterable[int]] = None,
    crd: Optional[Tuple[int, int, int, int]] = None,
    scale_factors: Optional[ScaleFactors_t] = None,
    overwrite: bool = False,
) -> SpatialData:
    """
    Combines specific channels within an image layer of a SpatialData object. 
    When given, nuc_channels are aggregated together, as are mem_channels.

    Parameters
    ----------
    sdata : SpatialData
        Spatial data object containing the image to be combined.
    img_layer : Optional[str], default=None
        The image layer in `sdata` to process. If not provided, the last image layer in `sdata` is used.
    output_layer : Optional[str]
        The name of the output layer where results will be stored. This must be specified.
    nuc_channels : Optional[int | Iterable[int]], default=None
        Specifies which channel(s) to consider as nuclear channels.
    mem_channels : Optional[int | Iterable[int]], default=None
        Specifies which channel(s) to consider as membrane channels.
    crd : Optional[Tuple[int, int, int, int]], default=None
        The coordinates specifying the region of the image to be processed. Defines the bounds (x_min, x_max, y_min, y_max).
    scale_factors
        Scale factors to apply for multiscale.
    overwrite : bool, default=False
        If True, overwrites the output layer if it already exists in `sdata`.

    Returns
    -------
    SpatialData
        The `sdata` object with the combined image added to the specified output layer.
        If nuc_channels and mem_channels is not None, the nuc channels will be at position 0 and the mem channel at position 1.

    Raises
    ------
    ValueError
        If `output_layer` is not provided.
        If no channels are specified for combining.
        If provided arrays are not 2D (c,y,x).

    Notes
    -----
    The function combines specified channels from a SpatialData object's image layer, creating a new image layer.
    The provided channels can be specified as nuclear or membrane channels. If coordinates (crd) are specified, only
    the region within those coordinates will be considered for the combination. The function handles 2D images and uses dask
    for potential out-of-core computation.

    Examples
    --------
    Sum nuclear channels 0 and 1, and keep membrane channel 2 as is from the image layer "raw_image":

    >>> sdata = combine(sdata, img_layer="raw_image", output_layer="combined_image", nuc_channels=[0,1], mem_channels=2)

    Sum only nuclear channels 0 and 1:

    >>> sdata = combine(sdata, img_layer="raw_image", output_layer="nuc_combined", nuc_channels=[0,1])
    """

    if img_layer is None:
        img_layer = [*sdata.images][-1]
        log.warning(
            f"No image layer specified. "
            f"Applying image processing on the last image layer '{img_layer}' of the provided SpatialData object."
        )

    if output_layer is None:
        raise ValueError("Please specify a name for the output layer.")

    # get spatial element
    se = _get_spatial_element(sdata, layer=img_layer)

    def _process_channels(
        channels: Optional[int | Iterable[int]],
        se: Union[SpatialImage, DataArray],
        crd: Optional[Tuple[int, int, int, int]],
    ) -> Array:
        channels = (
            list(channels)
            if isinstance(channels, Iterable) and not isinstance(channels, str)
            else [channels]
        )
        arr = se.isel(c=channels).data
        if len(arr.shape) != 3:
            raise ValueError(
                f"Array is of dimension {arr.shape}, currently only 2D images are supported."
            )
        if crd is not None:
            arr = arr[crd[2] : crd[3], crd[0] : crd[1]]
        arr = arr.sum(axis=0)
        arr = arr[None, ...]
        return arr

    if crd is not None:
        crd = _substract_translation_crd(se, crd)

    results = []

    if nuc_channels is not None:
        nuc_arr = _process_channels(nuc_channels, se, crd)
        results.append(nuc_arr)

    if mem_channels is not None:
        mem_arr = _process_channels(mem_channels, se, crd)
        results.append(mem_arr)

    if len(results) == 2:
        arr = da.concatenate(results, axis=0)
    elif len(results) == 1:
        arr = results[0]
    else:
        raise ValueError(
            f"Please specify either nuc_channels "
            f"(currently set to: {nuc_channels}) or mem_channels (currently set to {mem_channels})."
        )

    tx, ty = _get_translation(se)

    if crd is not None:
        tx = tx + crd[0]
        ty = ty + crd[2]

    translation = Translation([tx, ty], axes=("x", "y"))

    sdata = _add_image_layer(
        sdata,
        arr=arr,
        output_layer=output_layer,
        chunks=arr.chunksize,
        transformation=translation,
        scale_factors=scale_factors,
        overwrite=overwrite,
    )

    return sdata
