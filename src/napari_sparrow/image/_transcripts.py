import numpy as np
import spatialdata
from scipy.ndimage import gaussian_filter
from spatialdata.transformations import Translation, set_transformation


def transcript_density(
    sdata,
    points_layer="transcripts",
    name_x="x",
    name_y="y",
    scaling_factor=100,
    crd=None,
    output_layer: str = "transcript_density",
):
    ddf = sdata.points[points_layer]

    ddf[name_x] = ddf[name_x].round().astype(int)
    ddf[name_y] = ddf[name_y].round().astype(int)

    if crd:
        ddf = ddf.query(
            f"{crd[0]} <= {name_x} < {crd[1] } and {crd[2]} <= {name_y} < {crd[3] }"
        )

    counts_location_transcript = ddf.groupby([ name_x, name_y ]).count().compute()["gene"]
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


def transcript_density_deprecated(df, scaling_factor=100):  # TODO: add type hints
    """This function plots the transcript density of the tissue. You can use it to compare different regions in your tissue on transcript density."""
    Try = df.groupby(["x", "y"]).count()["gene"]  # TODO: rename Try + lower case
    print("grouping finished")
    image = np.array(Try.unstack(fill_value=0))
    print("unstack finished")
    image = image / np.max(image)
    print("starting gaussian filter")
    blurred = gaussian_filter(scaling_factor * image, sigma=7)
    return blurred.T
