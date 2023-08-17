from typing import Optional, Tuple

import numpy as np
import spatialdata
from scipy.ndimage import gaussian_filter
from spatialdata import SpatialData
from spatialdata.transformations import Translation, set_transformation


def transcript_density(
    sdata: SpatialData,
    points_layer: str = "transcripts",
    name_x: str = "x",
    name_y: str = "y",
    scaling_factor: float = 100,
    crd: Optional[Tuple[int, int, int, int]] = None,
    output_layer: str = "transcript_density",
)->SpatialData:
    """
    Calculate the transcript density and add it to the provided spatial data.

    This function computes the density of transcripts in the spatial data, scales and smooths it, 
    and then adds the resulting density image to the spatial data object.

    Parameters
    ----------
    sdata : SpatialData
        Data containing spatial information.
    points_layer : str, optional
        The layer name that contains the transcript data points, by default "transcripts".
    name_x : str, optional
        Column name for x-coordinates of the transcripts in the points layer, by default "x".
    name_y : str, optional
        Column name for y-coordinates of the transcripts in the points layer, by default "y".
    scaling_factor : float, optional
        Factor to scale the transcript density image, by default 100.
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

    if crd:
        ddf = ddf.query(
            f"{crd[0]} <= {name_x} < {crd[1] } and {crd[2]} <= {name_y} < {crd[3] }"
        )

    counts_location_transcript = ddf.groupby([name_x, name_y]).count().compute()["gene"]
    counts_location_transcript

    counts_location_transcript = counts_location_transcript.reset_index()

    if crd:
        counts_location_transcript[name_x] = counts_location_transcript[name_x] - crd[0]
        counts_location_transcript[name_y] = counts_location_transcript[name_y] - crd[2]

    counts_location_transcript = counts_location_transcript.set_index([name_x, name_y])

    image = np.array(counts_location_transcript.unstack(fill_value=0))

    image = image / np.max(image)
    blurred_transcripts = gaussian_filter(scaling_factor * image, sigma=7)
    blurred_transcripts = blurred_transcripts.T

    spatial_image = spatialdata.models.Image2DModel.parse(
        blurred_transcripts[None,], dims=("c", "y", "x")
    )

    if crd:
        translation = Translation([crd[0], crd[2]], axes=("x", "y"))
        set_transformation(spatial_image, translation)

    # during adding of image it is written to zarr store
    sdata.add_image(name=output_layer, image=spatial_image)

    return sdata