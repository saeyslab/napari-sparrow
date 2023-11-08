from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from dask.dataframe.core import DataFrame as DaskDataFrame
from spatialdata import SpatialData

from napari_sparrow.image._image import (
    _apply_transform,
    _get_boundary,
    _get_spatial_element,
    _unapply_transform,
)
from napari_sparrow.plot._plot import _get_z_slice_polygons
from napari_sparrow.shape import intersect_rectangles
from napari_sparrow.shape._shape import _extract_boundaries_from_geometry_collection
from napari_sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def sanity_plot_transcripts_matrix(
    sdata: SpatialData,
    img_layer: Optional[str] = None,
    labels_layer: Optional[str] = None,
    points_layer: str = "transcripts",
    shapes_layer: Optional[str] = None,
    channel: Optional[int | str] = None,
    z_slice: Optional[int] = None,
    plot_cell_number: bool = False,
    n_sample: Optional[int] = None,
    name_x: str = "x",
    name_y: str = "y",
    name_z: str = "z",
    name_gene_column: str = "gene",
    gene: Optional[str] = None,
    crd: Optional[Tuple[int, int, int, int]] = None,
    output: Optional[Path | str] = None,
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
    labels_layer : str or None, optional
        Labels layer to be plotted.
    points_layer : str, default="transcripts"
        The points layer in the SpatialData object representing transcripts.
    shapes_layer : Optional[str], default=None
        The layer in the SpatialData object representing cell boundaries. If None, no cell boundaries are plotted.
    channel : int or str or None, optional
        Channel to display from the img_layer. If none provided, or if provided channel could not be found, first channel is plot.
    z_slice: int or None, optional
        The z_slice to visualize in case of 3D (c,z,y,x) image/polygons.
        If z_slice is not specified and the sdata[image_layer] is three-dimensional, the plot defaults to the z-slice at index 0;
        shapes from all z-stacks are displayed; alongside transcripts from every z-stack.
    plot_cell_number : bool, default=False
        Whether to annotate cells with their numbers on the plot.
    n_sample : Optional[int], default=None
        The number of transcripts to sample for plotting. Useful for large datasets.
    name_x : str, default="x"
        Column name in the points_layer representing x-coordinates of transcripts.
    name_y : str, default="y"
        Column name in the points_layer representing y-coordinates of transcripts.
    name_z : str, default="z"
        Column name in the points_layer representing z-coordinates of transcripts.
    name_gene_column : str, default="gene"
        Column name in the points_layer representing gene information.
    gene : Optional[str], default=None
        Specific gene to filter and plot. If None, all genes are plotted.
    crd : Optional[Tuple[int, int, int, int]], default=None
        Coordinates to define a rectangular region for plotting as (xmin, xmax, ymin, ymax).
        If None, the entire image boundary is used.
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
    if img_layer is not None and labels_layer is not None:
        raise ValueError(
            "Both img_layer and labels_layer is not None. "
            "Please specify either img_layer or labels_layer, not both."
        )

    # Choose the appropriate layer or default to the last image layer if none is specified.
    if img_layer is not None:
        layer = img_layer
        img_layer_type = True
    elif labels_layer is not None:
        layer = labels_layer
        img_layer_type = False
    else:
        layer = [*sdata.images][-1]
        img_layer_type = True
        log.warning(
            f"No image layer or labels layer specified. "
            f"Plotting last image layer {layer} of the provided SpatialData object."
        )

    se = _get_spatial_element(sdata, layer=layer)

    _, ax = plt.subplots(figsize=(10, 10))

    image_boundary = _get_boundary(se)

    if crd is not None:
        _crd = crd
        crd = intersect_rectangles(crd, image_boundary)
        if crd is None:
            log.warning(
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

    se, x_coords_orig, y_coords_orig = _apply_transform(se)

    if img_layer_type:
        if channel is None:
            # if channel is None, plot the first channel
            channel = se.c.data[0]
            # if channel not in spatialelement, plot the first channel
        elif channel not in se.c.data:
            _channel = channel
            channel = se.c.data[0]
            log.warning(
                (
                    f"Provided channel '{_channel}' not in list of available channels '{se.c.data}' "
                    f"for provided img_layer '{layer}'. Falling back to plotting first available channel '{channel}' for this img_layer."
                )
            )

        channel_name = se.c.name
        channel_idx = list(se.c.data).index(channel)
        _se = se.isel(c=channel_idx)
        cmap_layer = "gray"
    else:
        _se = se
        cmap_layer = "viridis"

    if z_slice is not None:
        if _se.ndim == 3:
            _se = _se[z_slice, ...]
    else:
        if _se.ndim == 3:
            log.warning(
                f"Layer {layer} has 3 dimensions, but no z-slice was added. Using z_slice at index 0 for plotting by default."
            )
            _se = _se[0, ...]
        _se = _se.squeeze()

    _se.sel(x=slice(crd[0], crd[1]), y=slice(crd[2], crd[3])).plot.imshow(
        cmap=cmap_layer, robust=True, ax=ax, add_colorbar=False
    )

    se = _unapply_transform(se, x_coords_orig, y_coords_orig)

    if not hasattr(sdata, "points"):
        raise AttributeError("Please first read transcripts in SpatialData object.")

    in_df = sdata.points[points_layer]

    # query first and then slicing gene is faster than vice versa
    in_df = in_df.query(
        f"{crd[0]} <= {name_x} < {crd[1]} and {crd[2]} <= {name_y} < {crd[3]}"
    )

    if z_slice is not None:
        in_df = in_df.query(f"{name_z} == {z_slice}")

    if gene:
        in_df = in_df[in_df[name_gene_column] == gene]

    # we do not sample a fraction of the transcripts if a specific gene is given
    else:
        size = len(in_df)

        log.info(f"size before sampling is {size}")

        if n_sample is not None and size > n_sample:
            fraction = n_sample / size
            in_df = in_df.sample(frac=fraction)

    if isinstance(in_df, DaskDataFrame):
        in_df = in_df.compute()

    log.info(f"Plotting {in_df.shape[0]} transcripts.")

    if gene:
        alpha = 0.5
    else:
        alpha = 0.2

    ax.scatter(in_df[name_x], in_df[name_y], color="r", s=8, alpha=alpha)

    if shapes_layer is not None:
        polygons = sdata.shapes[shapes_layer]
    else:
        polygons = None

    if polygons is not None:
        log.info("Selecting boundaries")

        polygons = polygons.cx[crd[0] : crd[1], crd[2] : crd[3]]

        if z_slice is not None:
            polygons = _get_z_slice_polygons(polygons, z_slice=z_slice)

        if not polygons.empty:
            polygons["boundaries"] = polygons["geometry"].apply(
                _extract_boundaries_from_geometry_collection
            )
            exploded_boundaries = polygons.explode("boundaries")
            exploded_boundaries["geometry"] = exploded_boundaries["boundaries"]
            exploded_boundaries = exploded_boundaries.drop(columns=["boundaries"])

            log.info("Plotting boundaries")

            # Plot the polygon boundaries
            exploded_boundaries.plot(
                ax=ax,
                aspect=1,
            )

            log.info("End plotting boundaries")

            # Plot the values inside the polygons
            cell_numbers_plotted = set()
            if plot_cell_number:
                for _, row in polygons.iterrows():
                    centroid = row.geometry.centroid
                    value = row.name
                    if value in cell_numbers_plotted:
                        # plot cell number only once for 3D stack of polygons
                        continue
                    cell_numbers_plotted.add(value)
                    ax.annotate(
                        value,
                        (centroid.x, centroid.y),
                        color="green",
                        fontsize=20,
                        ha="center",
                        va="center",
                    )
        else:
            log.warning(f"Shapes layer {shapes_layer} was empty for crd {crd}.")

    ax.set_xlim(crd[0], crd[1])
    ax.set_ylim(crd[2], crd[3])

    ax.invert_yaxis()

    ax.axis("on")
    if img_layer_type:
        ax.set_title(f"{channel_name}={channel}")

    if gene:
        ax.set_title(f"Transcripts and cell boundaries for gene: {gene}.")

    if output:
        plt.savefig(output)
    else:
        plt.show()
    plt.close()
