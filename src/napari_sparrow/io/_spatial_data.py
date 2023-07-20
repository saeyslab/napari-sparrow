from typing import List, Optional, Union
from pathlib import Path
import warnings
import numpy as np
import dask.array as da
from dask_image import imread
import spatialdata
from spatialdata import bounding_box_query
from spatialdata import SpatialData
from spatialdata.transformations import get_transformation


def create_sdata(
    input: Union[str, Path, np.ndarray, List[Union[str, Path, np.ndarray]]],
    output_path: Optional[str | Path] = None,
    layer_name: str = "raw_image",
    chunks: Optional[int] = None,
    dims: Optional[List[str]] = None,
    crd: Optional[List[int]] = None,
):
    dask_array = _load_image_to_dask(input=input, chunks=chunks, dims=dims)

    # make sure we have ( c, z, y, x )
    if len(dask_array.shape) == 4:
        # put channel dimension first (dask image puts channel dim last)
        dask_array = dask_array.transpose(3, 0, 1, 2)
    elif len(dask_array.shape) == 3:
        dask_array = dask_array[None, :, :, :]
    else:
        raise ValueError(
            f"Image has dimension { dask_array.shape }, while (c, z, y, x) is required."
        )

    if chunks:
        dask_array = dask_array.rechunk(chunks)

    # perform z-projection
    if dask_array.shape[1] > 1:
        dask_array = da.max(dask_array, axis=1)

    # if z-dimension is 1, then squeeze it.
    else:
        dask_array = dask_array.squeeze(1)
    sdata = spatialdata.SpatialData()

    spatial_image = spatialdata.models.Image2DModel.parse(
        dask_array, dims=("c", "y", "x")
    )

    if crd and any(crd):
        for i, val in enumerate(crd):
            if val is None:
                if i == 0 or i == 2:  # x_min or y_min
                    crd[i] = 0
                elif i == 1:  # x_max
                    crd[i] = spatial_image.shape[2]
                elif i == 3:  # y_max
                    crd[i] = spatial_image.shape[1]

        spatial_image = bounding_box_query(
            spatial_image,
            axes=("x", "y"),
            min_coordinate=[crd[0], crd[2]],
            max_coordinate=[crd[1], crd[3]],
            target_coordinate_system="global",
        )
        if chunks:
            spatial_image = spatial_image.chunk(chunks)

        transform_matrix = get_transformation(spatial_image).to_affine_matrix(
            input_axes=("x", "y"), output_axes=("x", "y")
        )

    sdata.add_image(name=layer_name, image=spatial_image)

    if output_path is not None:
        sdata.write(output_path)

    return sdata


def _load_image_to_dask(
    input: Union[
        str, Path, np.ndarray, da.Array, List[Union[str, Path, np.ndarray, da.Array]]
    ],
    chunks: Optional[int] = None,
    dims: Optional[List[str]] = None,
    crd: Optional[List[int]] = None,
) -> da.Array:
    """
    Load images into a dask array.

    This function facilitates the loading of one or more images into a 3D dask array.
    These images are designated by a provided filename pattern or numpy array. The resulting dask
    array will have three dimensions structured as follows: channel (c), y, and x.

    The input parameter can be formatted in four ways:

    - A path to a single image, either grayscale or multiple channels.
        Examples:
        DAPI_z3.tif -> single channel
        DAPI_Poly_z3.tif -> multi (DAPI, Poly) channel
    - A pattern representing a collection of z-stacks (if this is the case, a z-projection
    is performed which selects the maximum intensity value across the z-dimension).
        Examples:
        DAPI_z*.tif -> z-projection performed
        DAPI_Poly_z*.tif -> z-projection performed
    - A list of filename patterns (where each list item corresponds to a different channel)
        Examples
        [ DAPI_z3.tif, Poly_z3.tif ] -> multi (DAPI, Poly) channel
        [ DAPI_z*.tif, Poly_z*.tif ] -> multi (DAPI, Poly) channel, z projection performed
    -A single numpy or dask array. The dims parameter should specify its dimension, e.g. ['y','x']
    for a 2D array or [ 'c', 'y', 'x', 'z' ] for a 4D array with channels.
    If a z-dimension >1 is present, a z-projection is performed.

    Parameters
    ----------
    filename_pattern : Union[str, Path, List[Union[str, Path]]]
        The filename pattern, path or list of filename patterns to the images that
        should be loaded. In case of a list, each list item should represent a different
        channel, and each image corresponding to a filename pattern should represent
        a different z-stack.
    chunks : int, optional
        Chunk size for rechunking the resulting dask array. If not provided (None),
        the array will not be rechunked.
    dims : int, optional
        Dimension of the provided array. If input is a str of Path, this parameter is ignored.

    Returns
    -------
    dask.array.Array
        The resulting 3D dask array with dimensions ordered as: (c, y, x).

    Raises
    ------
    ValueError
        If an image is not 3D (z,y,x), a ValueError is raised. z-dimension can be 1.
    """

    if isinstance(input, list):
        # if filename pattern is a list, create (c, y, x) for each filename pattern
        dask_arrays = [_load_image_to_dask(f, chunks, dims) for f in input]
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
            warnings.warn(
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

    return dask_array


def _fix_dimensions(
    array: np.ndarray,
    dims: List[str] = ["c", "z", "y", "x"],
    target_dims: List[str] = ["c", "z", "y", "x"],
):
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
