import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import spatialdata
from dask.array import from_zarr
from dask_image import imread
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import Translation

from napari_sparrow.image._image import _add_image_layer
from napari_sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def create_sdata(
    input: str
    | Path
    | np.ndarray
    | da.Array
    | List[str]
    | List[Path]
    | List[np.ndarray]
    | List[da.Array],
    output_path: Optional[str | Path] = None,
    img_layer: str = "raw_image",
    chunks: Optional[str | tuple[int, int, int, int] | int] = None,
    dims: Optional[List[str]] = None,
    crd: Optional[Tuple[int, int, int, int]] = None,
    scale_factors: Optional[ScaleFactors_t] = None,
) -> SpatialData:
    """
    Convert input images or arrays into a SpatialData object.

    This function allows you to ingest various input formats of images or data arrays,
    convert them into a unified SpatialData format and write them out to a specified
    path (zarr store) if needed. It provides flexibility in how you define the source data as well
    as certain processing options like chunking.

    The input parameter can be formatted in four ways:

    - Path to a single image, either grayscale or multiple channels.

      Examples:

      input=DAPI_z3.tif -> single channel

      input=DAPI_Poly_z3.tif -> multi (DAPI, Poly) channel

    - Pattern representing a collection of z-stacks 
      (if this is the case, a z-projection is performed which selects the maximum intensity value across the z-dimension).

      Examples:

      input=DAPI_z*.tif -> z-projection performed

      input=DAPI_Poly_z*.tif -> z-projection performed

    - List of filename patterns (where each list item corresponds to a different channel)

      Examples:

      input=[ DAPI_z3.tif, Poly_z3.tif ] -> multi (DAPI, Poly) channel

      input[ DAPI_z*.tif, Poly_z*.tif ] -> multi (DAPI, Poly) channel, z projection performed

    - Single numpy or dask array. 
      The dims parameter should specify its dimension, e.g. ['y','x'] for a 2D array or 
      [ 'c', 'y', 'x', 'z' ] for a 4D array with channels. If a z-dimension >1 is present, a z-projection is performed.

    Parameters
    ----------
    input : Union[str, Path, List[Union[str, Path]]]
        The filename pattern, path or list of filename patterns to the images that
        should be loaded. In case of a list, each list item should represent a different
        channel, and each image corresponding to a filename pattern should represent.
        a different z-stack.
        Input can also be a numpy array. In that case the dims parameter should be specified.
    output_path : Optional[str | Path], default=None
        If specified, the resulting SpatialData object will be written to this path as a zarr.
    img_layer : str, default="raw_image"
        The name of the image layer to be created in the SpatialData object.
    chunks : Optional[int, tuple], default=None
        If specified, the underlying dask array will be rechunked to this size.
        If Tuple, desired chunksize along c,z,y,x should be specified, e.g. (1,1,1024,1024).
    dims : Optional[List[str]], default=None
        The dimensions of the input data if it's a numpy array. E.g., ['y','x'] or ['c','y','x','z'].
        If input is a str, Path or List[str], List[Path], this parameter is ignored.
    crd : tuple of int, optional
        The coordinates for a region of interest in the format (xmin, xmax, ymin, ymax).
        If specified, this region is cropped from the image, and added as image layer to the
        SpatialData object.
    scale_factors
        Scale factors to apply for multiscale.

    Returns
    -------
    SpatialData
        The constructed SpatialData object containing image layer with name 'img_layer' with dimension (c,y,x)

    Notes
    -----
    If 'crd' is specified and some of its values are None, the function infers the missing
    values based on the input image's shape.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        # tmp_dir is used to write results for each channel out to separate zarr store,
        # these results are then combined and written to sdata
        dask_array = _load_image_to_dask(
            input=input, chunks=chunks, dims=dims, crd=crd, output_dir=tmp_dir
        )

        sdata = spatialdata.SpatialData()

        # make sure sdata is backed.
        if output_path is not None:
            sdata.write(output_path)

        if crd is not None:
            crd = _fix_crd(crd, dask_array)
            tx = crd[0]
            ty = crd[2]

            translation = Translation([tx, ty], axes=("x", "y"))

        else:
            translation = None

        _add_image_layer(
            sdata,
            arr=dask_array,
            output_layer=img_layer,
            chunks=chunks,
            transformation=translation,
            scale_factors=scale_factors,
            overwrite=False,
        )

    return sdata


def _load_image_to_dask(
    input: str
    | Path
    | np.ndarray
    | da.Array
    | List[str]
    | List[Path]
    | List[np.ndarray]
    | List[da.Array],
    chunks: str | Tuple[int, int, int, int] | int | None = None,
    dims: Optional[List[str]] = None,
    crd: Optional[List[int]] = None,
    output_dir: Optional[Union[Path, str]] = None,
) -> da.Array:
    """
    Load images into a dask array.

    This function facilitates the loading of one or more images into a 3D dask array.
    These images are designated by a provided filename pattern or numpy array. The resulting dask
    array will have three dimensions structured as follows: channel (c), y, and x.

    Parameters
    ----------
    input : Union[str, Path, List[Union[str, Path]]]
        The filename pattern, path or list of filename patterns to the images that
        should be loaded. In case of a list, each list item should represent a different
        channel, and each image corresponding to a filename pattern should represent.
        a different z-stack.
    chunks : Optional[int], default=None
        If specified, the underlying dask array will be rechunked to this size.
    dims : Optional[List[str]], default=None
        The dimensions of the input data if it's a numpy array. E.g., ['y','x'] or ['c','y','x','z'].
        If input is a str of Path, this parameter is ignored.
    crd : tuple of int, optional
        The coordinates for a region of interest in the format (xmin, xmax, ymin, ymax).
    output_dir: if input is a str, Path, or List[str], List[Path], the intermediate results for each channel
        will be written to zarr store in output_dir to prevent unnecessary memory usage.

    Returns
    -------
    dask.array.Array
        The resulting 3D dask array with dimensions ordered as: (c, y, x).

    Raises
    ------
    ValueError
        If an image is not 3D (z,y,x) after loading by dask_image, a ValueError is raised. z-dimension can be 1.
    """

    if isinstance(input, list):
        # if filename pattern is a list, create (c, y, x) for each filename pattern
        dask_arrays = [
            _load_image_to_dask(f, chunks, dims, output_dir=output_dir) for f in input
        ]

        dask_array = da.concatenate(dask_arrays, axis=0)
        # add z- dimension, we want (c,z,y,x)
        dask_array = dask_array[:, None, :, :]

    elif isinstance(input, (np.ndarray, da.Array)):
        # make sure we have (c,z,y,x)
        array = _fix_dimensions(input, dims=dims)
        if isinstance(array, np.ndarray):
            dask_array = da.from_array(array)
        else:
            dask_array = array

    elif isinstance(input, (str, Path)):
        if dims is not None:
            log.warning(
                (
                    f"dims parameter is equal to {dims}, but will be ignored when reading in images from a file"
                )
            )
        # make sure we have (c,z,y,x)
        dask_array = imread.imread(input)
        if len(dask_array.shape) == 4:
            # dask_image puts channel dim last
            dims = ["z", "y", "x", "c"]
            dask_array = _fix_dimensions(dask_array, dims=dims)
        elif len(dask_array.shape) == 3:
            # dask_image does not add channel dim for grayscale images
            dims = ["z", "y", "x"]
            dask_array = _fix_dimensions(dask_array, dims=dims)
        elif len(dask_array.shaep) == 2:
            dims = ["y", "x"]
            dask_array = _fix_dimensions(dask_array, dims=dims)
        else:
            raise ValueError(
                f"Image has shape { dask_array.shape }, while (y, x) is required."
            )
    else:
        raise ValueError(f"input of type {type(input)} not supported.")

    if crd:
        # c,z,y,x
        # get slice of it:
        dask_array = dask_array[:, :, crd[2] : crd[3], crd[0] : crd[1]]

    if chunks:
        dask_array = dask_array.rechunk(chunks)

    # perform z-projection
    if dask_array.shape[1] > 1:
        dask_array = da.max(dask_array, axis=1)
    # if z-dimension is 1, then squeeze it.
    else:
        dask_array = dask_array.squeeze(1)

    if isinstance(input, (str, Path)) and output_dir is not None:
        name = os.path.splitext(os.path.basename(input))[0]
        output_zarr = os.path.join(output_dir, f"{name}.zarr")
        dask_array.to_zarr(output_zarr)
        dask_array = from_zarr(output_zarr)

    return dask_array


def _fix_dimensions(
    array: np.ndarray | da.Array,
    dims: List[str] = ["c", "z", "y", "x"],
    target_dims: List[str] = ["c", "z", "y", "x"],
) -> np.ndarray | da.Array:
    dims = list(dims)
    target_dims = list(target_dims)

    dims_set = set(dims)
    if len(dims) != len(dims_set):
        raise ValueError(f"dims list {dims} contains duplicates")

    target_dims_set = set(target_dims)

    extra_dims = dims_set - target_dims_set

    if extra_dims:
        raise ValueError(
            f"The dimension(s) {extra_dims} are not present in the target dimensions."
        )

    # check if the array already has the correct number of dimensions, if not add missing dimensions
    if len(array.shape) != len(dims):
        raise ValueError(
            f"Dimension of array {array.shape} is not equal to dimension of provided dims { dims}"
        )
    if len(array.shape) > 4:
        raise ValueError(
            f"Arrays with dimension larger than 4 are not supported, shape of array is {array.shape}"
        )

    for dim in target_dims:
        if dim not in dims:
            array = array[None, ...]
            dims.insert(0, dim)

    # create a mapping from input dims to target dims
    dim_mapping = {dim: i for i, dim in enumerate(dims)}
    target_order = [dim_mapping[dim] for dim in target_dims]

    # transpose the array to the target dimension order
    array = array.transpose(*target_order)

    return array


def _fix_crd(
    crd: Tuple[int, int, int, int], arr: da.Array
) -> Tuple[int, int, int, int]:
    x1, x2, y1, y2 = crd

    # dask array has dimension c,y,x
    # Get the shape of the array
    _, max_y, max_x = arr.shape

    # Update the coordinates based on the conditions
    x1 = 0 if x1 is None else x1
    y1 = 0 if y1 is None else y1
    x2 = max_x if x2 is None else x2
    y2 = max_y if y2 is None else y2

    return (x1, x2, y1, y2)
