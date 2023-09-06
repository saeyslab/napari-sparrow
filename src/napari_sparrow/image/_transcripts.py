import warnings
from typing import Optional, Tuple

import dask.array as da
import spatialdata
from scipy.ndimage import gaussian_filter
from spatialdata import SpatialData
from spatialdata.transformations import Translation, set_transformation

from napari_sparrow.image._image import _get_image_boundary
from napari_sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def transcript_density(
    sdata: SpatialData,
    img_layer: Optional[str] = "raw_image",
    points_layer: str = "transcripts",
    n_sample: Optional[int] = 15000000,
    name_x: str = "x",
    name_y: str = "y",
    name_gene_column: str = "gene",
    scaling_factor: float = 100,
    chunks: int = 1024,
    crd: Optional[Tuple[int, int, int, int]] = None,
    output_layer: str = "transcript_density",
) -> SpatialData:
    """
    Calculate the transcript density using gaussian filter and add it to the provided spatial data.

    This function computes the density of transcripts in the spatial data, scales and smooths it,
    and then adds the resulting density image to the spatial data object.

    Parameters
    ----------
    sdata : SpatialData
        Data containing spatial information.
    img_layer : Optional[str], default=None
        The layer of the SpatialData object used for determining image boundary. Ignored if crd is specified.
        Defaults to the last layer if not provided.
    points_layer : str, optional
        The layer name that contains the transcript data points, by default "transcripts".
    n_sample : Optional[int], default=15000000
        The number of transcripts to sample for calculation of transcript density.
    name_x : str, optional
        Column name for x-coordinates of the transcripts in the points layer, by default "x".
    name_y : str, optional
        Column name for y-coordinates of the transcripts in the points layer, by default "y".
    name_gene_column : str, optional
        Column name in the points_layer representing gene information, by default "gene".
    scaling_factor : float, optional
        Factor to scale the transcript density image, by default 100.
    chunks: int.
        Chunksize for calculation of density using gaussian filter.
    crd : tuple of int, optional
        The coordinates for a region of interest in the format (xmin, xmax, ymin, ymax).
        If provided, the density is computed only for this region, by default None.
    output_layer : str, optional
        The name of the output image layer in the SpatialData where the transcript density will be added,
        by default "transcript_density".

    Returns
    -------
    SpatialData
        Updated spatial data object with the added transcript density layer as an image layer.

    Examples
    --------
    >>> sdata = SpatialData(...)
    >>> sdata = transcript_density(sdata, points_layer="transcripts", crd=(2000, 4000, 2000, 4000))

    """
    ddf = sdata.points[points_layer]

    ddf[name_x] = ddf[name_x].round().astype(int)
    ddf[name_y] = ddf[name_y].round().astype(int)

    # get image boundary from last image layer if img_layer is None
    if img_layer is None:
        img_layer = [*sdata.images][-1]
    img_boundary = _get_image_boundary(sdata[img_layer])

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
    blurred_transcripts=blurred_transcripts.rechunk( blurred_transcripts.chunksize )

    spatial_image = spatialdata.models.Image2DModel.parse(
        blurred_transcripts[None,], dims=("c", "y", "x")
    )

    if crd:
        translation = Translation([crd[0], crd[2]], axes=("x", "y"))
        set_transformation(spatial_image, translation)

    # during adding of image it is written to zarr store
    sdata.add_image(name=output_layer, image=spatial_image)

    return sdata
