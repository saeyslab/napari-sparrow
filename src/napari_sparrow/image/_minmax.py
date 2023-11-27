from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import dask.array as da
from dask.array import Array
from dask_image.ndfilters import maximum_filter, minimum_filter
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t

from napari_sparrow.image._apply import apply, _get_spatial_element

import numpy as np

from napari_sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def min_max_filtering(
    sdata: SpatialData,
    img_layer: Optional[str] = None,
    size_min_max_filter: int | List[int] = 85,
    output_layer="min_max_filtered",
    crd: Optional[Tuple[int, int, int, int]] = None,
    scale_factors: Optional[ScaleFactors_t] = None,
    overwrite: bool = False,
) -> SpatialData:
    """
    Apply min max filtering to an image in a SpatialData object using dask.
    The size of the filter can be provided
    either as an integer or a list of integers corresponding to each channel.
    Compatibility with image layers that have either two or three spatial dimensions.

    Parameters
    ----------
    sdata : SpatialData
        Spatial data object containing the images to be processed.
    img_layer : Optional[str], default=None
        The image layer in `sdata` to run min_max_filtering on. If not provided, the last image layer in `sdata` is used.
    size_min_max_filter : Union[int, List[int]], optional
        Size of the min_max filter. If provided as a list, the length
        must match the number of channels. Defaults to 85.
    output_layer : str, optional
        The name of the output layer. Defaults to "min_max_filtered".
    crd : Optional[Tuple[int, int, int, int]], default=None
        The coordinates specifying the region of the image to be processed. Defines the bounds (x_min, x_max, y_min, y_max).
    scale_factors
        Scale factors to apply for multiscale.
    overwrite: bool
        If True overwrites the element if it already exists.

    Returns
    -------
    SpatialData
        The `sdata` object with the min_max-filtered image added.

    Raises
    ------
    ValueError
        If `size_min_max_filter` is a list and its length does not match the number of channels.

    Examples
    --------
    Apply min max filtering with a single size:

    >>> sdata = min_max_filtering(sdata, size_min_max_filtering=50)

    Apply min max filtering with different sizes for each channel:

    >>> sdata = min_max_filtering(sdata, size_min_max_filtering=[30, 50, 70])
    """

    def _apply_min_max_filtering(image: Array, size_min_max_filter: int = 85) -> Array:

        def _to_odd( size_min_max_filter ):
            if not isinstance( size_min_max_filter, int ):
                log.warning("Non-integer value received for size_min_max_filter; it will be rounded to the nearest integer.")
                size_min_max_filter=int(np.round( size_min_max_filter ))
            if size_min_max_filter %2 == 0:
                log.warning( f"Provided value for min max filter size is even ('{size_min_max_filter}'). "
                         f"To prevent unexpected output, we set min max filter to '{size_min_max_filter +1}'." )
                return size_min_max_filter + 1
            else:
                return size_min_max_filter
            
        size_min_max_filter=_to_odd( size_min_max_filter )

        image_dim = image.ndim
        if image_dim == 3:
            if image.shape[0] == 1:
                image = da.squeeze(image, axis=0)
            else:
                raise ValueError(
                    "_apply_min_max_filtering only accepts c dimension equal to 1."
                )
        elif image_dim == 4:
            if image.shape[0] == 1 and image.shape[1] == 1:
                image = da.squeeze(image, axis=(0, 1))
            else:
                raise ValueError(
                    "_apply_min_max_filtering only accepts c and z dimension equal to 1."
                )
        else:
            raise ValueError(
                "Please provide numpy array containing c,(z),y and x dimension."
            )

        # Apply the minimum filter
        minimum_t = minimum_filter(image, size_min_max_filter)

        # Apply the maximum filter
        max_of_min_t = maximum_filter(minimum_t, size_min_max_filter)

        image = image - max_of_min_t

        if image_dim == 3:
            image = image[None, ...]
        else:
            image = image[None, None, ...]

        return image

    se = _get_spatial_element(sdata, img_layer)

    if isinstance(size_min_max_filter, Iterable):
        assert len(size_min_max_filter) == len(
            se.c.data
        ), f"If 'size_min_max_filter' is provided as a list, it should match the number of channels in '{se}' ({len(se.c.data)})"
        fn_kwargs = {
            key: {"size_min_max_filter": value}
            for (key, value) in zip(se.c.data, size_min_max_filter)
        }
    else:
        fn_kwargs = {"size_min_max_filter": size_min_max_filter}

    sdata = apply(
        sdata,
        _apply_min_max_filtering,
        fn_kwargs=fn_kwargs,
        img_layer=img_layer,
        output_layer=output_layer,
        combine_c=False,  # you want to process all channels independently.
        combine_z=False,  # you want to process all z-stacks independently.
        chunks=None,
        crd=crd,
        scale_factors=scale_factors,
        overwrite=overwrite,
    )

    return sdata
