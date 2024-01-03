import warnings
from typing import Optional, Tuple

import dask.array as da
from scipy.ndimage import gaussian_filter
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import Translation

from sparrow.image._image import _get_boundary, _get_spatial_element, _add_image_layer
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def transcript_density(
    sdata: SpatialData,
    img_layer: Optional[str] = "raw_image",
    points_layer: str = "transcripts",
    n_sample: Optional[int] = 15000000,
    name_x: str = "x",
    name_y: str = "y",
    name_z: Optional[str] = None,
    name_gene_column: str = "gene",
    z_index: Optional[int] = None,
    scaling_factor: float = 100,
    chunks: int = 1024,
    crd: Optional[Tuple[int, int, int, int]] = None,
    scale_factors: Optional[ScaleFactors_t] = None,
    output_layer: str = "transcript_density",
    overwrite: bool = False,
) -> SpatialData:
    """
    Calculate the transcript density using gaussian filter and add it to the provided spatial data.

    This function computes the density of transcripts in the spatial data, scales and smooths it,
    and then adds the resulting density image to the spatial data object.

    Parameters
    ----------
    sdata : SpatialData
        Data containing spatial information.
    img_layer : Optional[str], default="raw_image"
        The layer of the SpatialData object used for determining image boundary.
        Defaults to the last layer if set to None.
    points_layer : str, optional
        The layer name that contains the transcript data points, by default "transcripts".
    n_sample : Optional[int], default=15000000
        The number of transcripts to sample for calculation of transcript density.
    name_x : str, optional
        Column name for x-coordinates of the transcripts in the points layer, by default "x".
    name_y : str, optional
        Column name for y-coordinates of the transcripts in the points layer, by default "y".
    name_z : str or None, optional
        Column name for z-coordinates of the transcripts in the points layer, by default None.
    name_gene_column : str, optional
        Column name in the points_layer representing gene information, by default "gene".
    z_index: int or None, optional
        The z index in the points layer for which to calculate transcript density. If set to None for a 3D points layer 
        (and `name_z` is not equal to None), an y-x transcript density projection will be calculated.
    scaling_factor : float, optional
        Factor to scale the transcript density image, by default 100.
    chunks: int.
        Chunksize for calculation of density using gaussian filter.
    crd : tuple of int, optional
        The coordinates for a region of interest in the format (xmin, xmax, ymin, ymax).
        If provided, the density is computed only for this region, by default None.
    scale_factors
        Scale factors to apply for multiscale.
    output_layer : str, optional
        The name of the output image layer in the SpatialData where the transcript density will be added,
        by default "transcript_density".
    overwrite: bool
        If True overwrites the element if it already exists.

    Returns
    -------
    SpatialData
        Updated spatial data object with the added transcript density layer as an image layer.

    Examples
    --------
    >>> sdata = SpatialData(...)
    >>> sdata = transcript_density(sdata, points_layer="transcripts", crd=(2000, 4000, 2000, 4000))

    """
    if z_index is not None and name_z is None:
        raise ValueError( "Please specify column name for the z-coordinates of the transcripts in the points layer "
                         "when specifying z_index." )

    ddf = sdata.points[points_layer]

    ddf[name_x] = ddf[name_x].round().astype(int)
    ddf[name_y] = ddf[name_y].round().astype(int)
    if name_z is not None:
        ddf[name_z] = ddf[name_z].round().astype(int)

    # get image boundary from last image layer if img_layer is None
    if img_layer is None:
        img_layer = [*sdata.images][-1]

    se = _get_spatial_element(sdata, layer=img_layer)

    img_boundary = _get_boundary(se)

    # if crd is None, get boundary from image at img_layer if given,
    if crd is None:
        crd = img_boundary
    else:
        # fix crd so it falls in boundaries of img_layer, otherwise possible issues with registration transcripts, and size of the generated image.
        _crd = [
            max(img_boundary[0], crd[0]),
            min(img_boundary[1], crd[1]),
            max(img_boundary[2], crd[2]),
            min(img_boundary[3], crd[3]),
        ]
        if _crd != crd:
            warnings.warn(
                (
                    f"Provided crd didn't fully fit within the image layer '{img_layer}' with image boundary '{img_boundary}'. "
                    f"The crd was updated from '{crd}' to '{_crd}'."
                )
            )
        crd = _crd
    ddf = ddf.query(
        f"{crd[0]} <= {name_x} < {crd[1] } and {crd[2]} <= {name_y} < {crd[3] }"
    )

    if z_index is not None:
        ddf = ddf.query(f"{name_z} == {z_index}")

    # subsampling:
    if n_sample is not None:
        size = len(ddf)
        if size > n_sample:
            log.info(
                f"The number of transcripts ( {size} ) is larger than n_sample, sampling {n_sample} transcripts."
            )
            fraction = n_sample / size
            ddf = ddf.sample(frac=fraction)
            log.info("sampling finished")

    counts_location_transcript = ddf.groupby([name_x, name_y]).count()[name_gene_column]

    counts_location_transcript = counts_location_transcript.reset_index()

    if crd:
        counts_location_transcript[name_x] = counts_location_transcript[name_x] - crd[0]
        counts_location_transcript[name_y] = counts_location_transcript[name_y] - crd[2]

    chunks = (chunks, chunks)
    image = da.zeros((crd[1] - crd[0], crd[3] - crd[2]), chunks=chunks, dtype=int)

    def populate_chunk(block, block_info=None, x=None, y=None, values=None):
        # Extract the indices of the current block
        x_start, x_stop = block_info[0]["array-location"][0]
        y_start, y_stop = block_info[0]["array-location"][1]

        # Find the overlapping indices
        mask = (x >= x_start) & (x < x_stop) & (y >= y_start) & (y < y_stop)
        relevant_x = x[mask] - x_start
        relevant_y = y[mask] - y_start
        relevant_values = values[mask]

        # Populate the block + create copy of the block, so we can modify it.
        block_copy = block.copy()

        block_copy[relevant_x, relevant_y] = relevant_values
        return block_copy

    x_values = counts_location_transcript[name_x].values
    y_values = counts_location_transcript[name_y].values
    gene_values = counts_location_transcript[name_gene_column].values

    image = image.map_blocks(
        populate_chunk, x=x_values, y=y_values, values=gene_values, dtype=int
    )

    image = scaling_factor * (image / da.max(image))

    sigma = 7

    def chunked_gaussian_filter(chunk):
        return gaussian_filter(chunk, sigma=sigma)

    # take overlap to be 3 times sigma
    overlap = sigma * 3

    blurred_transcripts = image.map_overlap(
        chunked_gaussian_filter, depth=overlap, boundary="reflect"
    )

    blurred_transcripts = blurred_transcripts.T
    # rechunk, otherwise possible issues when saving to zarr
    blurred_transcripts = blurred_transcripts.rechunk(blurred_transcripts.chunksize)

    if crd:
        translation = Translation([crd[0], crd[2]], axes=("x", "y"))
    else:
        translation=None

    arr=blurred_transcripts[None,]

    sdata=_add_image_layer(
        sdata,
        arr=arr,
        output_layer=output_layer,
        chunks=arr.chunksize,
        transformation=translation,
        scale_factors=scale_factors,
        overwrite=overwrite,
    )

    return sdata
