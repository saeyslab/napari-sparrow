import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
from dask.dataframe.core import DataFrame as DaskDataFrame
from spatialdata import SpatialData

from napari_sparrow.image._image import (
    _apply_transform,
    _get_image_boundary,
    _unapply_transform,
)
from napari_sparrow.shape import intersect_rectangles
from napari_sparrow.shape._shape import _extract_boundaries_from_geometry_collection


def sanity_plot_transcripts_matrix(
    sdata: SpatialData,
    img_layer: Optional[str] = None,
    points_layer: str = "transcripts",
    shapes_layer: Optional[str] = None,
    plot_cell_number: bool = False,
    n_sample: Optional[int] = None,
    name_x: str = "x",
    name_y: str = "y",
    name_gene_column: str = "gene",
    gene: Optional[str] = None,
    crd: Optional[Tuple[int, int, int, int]] = None,
    cmap: str = "gray",
    output: Optional[Union[Path, str]] = None,
) -> None:
    """
    Produce a sanity plot to visualize spatial transcriptomics data on top of an image.

    This function plots a spatial image (e.g. microscopy or histology image) and overlays transcripts
    and optionally cell boundaries to visually inspect the spatial alignment of the data.
    This can be particularly useful to check data registration and alignment in spatial transcriptomics.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object containing the spatial image and transcripts data.
    img_layer : Optional[str], default=None
        The layer of the SpatialData object to be plotted. Defaults to the last layer if not provided.
    points_layer : str, default="transcripts"
        The points layer in the SpatialData object representing transcripts.
    shapes_layer : Optional[str], default=None
        The layer in the SpatialData object representing cell boundaries. If None, no cell boundaries are plotted.
    plot_cell_number : bool, default=False
        Whether to annotate cells with their numbers on the plot.
    n_sample : Optional[int], default=None
        The number of transcripts to sample for plotting. Useful for large datasets.
    name_x : str, default="x"
        Column name in the points_layer representing x-coordinates of transcripts.
    name_y : str, default="y"
        Column name in the points_layer representing y-coordinates of transcripts.
    name_gene_column : str, default="gene"
        Column name in the points_layer representing gene information.
    gene : Optional[str], default=None
        Specific gene to filter and plot. If None, all genes are plotted.
    crd : Optional[Tuple[int, int, int, int]], default=None
        Coordinates to define a rectangular region for plotting as (xmin, xmax, ymin, ymax).
        If None, the entire image boundary is used.
    cmap : str, default="gray"
        Colormap for displaying the image.
    output : Optional[Union[Path, str]], default=None
        Filepath to save the generated plot. If not provided, the plot will be displayed using plt.show().

    Returns
    -------
    None
        Either displays the plot or saves it to the specified output location.

    Raises
    ------
    AttributeError
        If the "points" attribute is not present in the SpatialData object.
    Warning
        If provided coordinates (crd) and image_boundary do not have overlap.
        If provided shapes_layer is not present in the SpatialData object.

    Notes
    -----
    - If a specific gene is provided, only transcripts corresponding to that gene are plotted.
    - If `plot_cell_number` is set to True, cells are annotated with their numbers.
    - If the `output` parameter is provided, the plot is saved to the specified location, otherwise, it's displayed.
    """
    if img_layer is None:
        img_layer = [*sdata.images][-1]

    xarray = sdata[img_layer]

    # plot for registration sanity check

    _, ax = plt.subplots(figsize=(10, 10))

    image_boundary = _get_image_boundary(xarray)

    if crd is not None:
        _crd = crd
        crd = intersect_rectangles(crd, image_boundary)
        if crd is None:
            warnings.warn(
                (
                    f"Provided crd '{_crd}' and image_boundary '{image_boundary}' do not have any overlap. "
                    f"Please provide a crd that has some overlap with the image. "
                    f"Setting crd to image_boundary '{image_boundary}'."
                )
            )
            crd = image_boundary
    # if crd is None, set crd equal to image_boundary
    else:
        crd = image_boundary

    xarray, x_coords_orig, y_coords_orig = _apply_transform(xarray)

    xarray.squeeze().sel(x=slice(crd[0], crd[1]), y=slice(crd[2], crd[3])).plot.imshow(
        cmap=cmap, robust=True, ax=ax, add_colorbar=False
    )

    xarray = _unapply_transform(xarray, x_coords_orig, y_coords_orig)

    if not hasattr(sdata, "points"):
        raise AttributeError("Please first read transcripts in SpatialData object.")

    in_df = sdata.points[points_layer]

    # query first and then slicing gene is faster than vice versa
    in_df = in_df.query(
        f"{crd[0]} <= {name_x} < {crd[1]} and {crd[2]} <= {name_y} < {crd[3]}"
    )

    if gene:
        in_df = in_df[in_df[name_gene_column] == gene]

    # we do not sample a fraction of the transcripts if a specific gene is given
    else:
        size = len(in_df)

        print(f"size before sampling is {size}")

        if n_sample is not None and size > n_sample:
            fraction = n_sample / size
            in_df = in_df.sample(frac=fraction)

    if isinstance(in_df, DaskDataFrame):
        in_df = in_df.compute()

    print(f"Plotting {in_df.shape[0]} transcripts.")

    if gene:
        alpha = 0.5
    else:
        alpha = 0.2

    ax.scatter(in_df[name_x], in_df[name_y], color="r", s=8, alpha=alpha)

    # we want to plot shapes if there is a shapes layer in the SpatialData object
    if hasattr(sdata, "shapes"):
        if shapes_layer is None:
            shapes_layer = [*sdata.shapes][-1]
            polygons = sdata.shapes[shapes_layer]
        elif shapes_layer in sdata.shapes:
            polygons = sdata[shapes_layer]
        else:
            _shapes_layer = shapes_layer
            shapes_layer = [*sdata.shapes][-1]
            warnings.warn(
                (
                    f"Provided shapes_layer '{_shapes_layer}' not in SpatialData object, plotting last shapes_layer '{shapes_layer}'."
                )
            )
            polygons = sdata.shapes[shapes_layer]
    else:
        polygons = None

    if polygons is not None:
        print("Selecting boundaries")

        polygons_selected = polygons.cx[crd[0] : crd[1], crd[2] : crd[3]]

        polygons_selected["boundaries"] = polygons_selected["geometry"].apply(
            _extract_boundaries_from_geometry_collection
        )
        exploded_boundaries = polygons_selected.explode("boundaries")
        exploded_boundaries["geometry"] = exploded_boundaries["boundaries"]
        exploded_boundaries = exploded_boundaries.drop(columns=["boundaries"])

        print("Plotting boundaries")

        # Plot the polygon boundaries
        exploded_boundaries.plot(
            ax=ax,
            aspect=1,
        )

        print("End plotting boundaries")

        # Plot the values inside the polygons
        if plot_cell_number:
            for _, row in polygons_selected.iterrows():
                centroid = row.geometry.centroid
                value = row.name
                ax.annotate(
                    value,
                    (centroid.x, centroid.y),
                    color="green",
                    fontsize=20,
                    ha="center",
                    va="center",
                )

    ax.set_xlim(crd[0], crd[1])
    ax.set_ylim(crd[2], crd[3])

    ax.invert_yaxis()

    ax.axis("on")

    if gene:
        ax.set_title(f"Transcripts and cell boundaries for gene: {gene}.")

    if output:
        plt.savefig(output)
    else:
        plt.show()
    plt.close()
