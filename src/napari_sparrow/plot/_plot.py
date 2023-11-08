from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from geopandas.geoseries import GeoSeries
from geopandas.geodataframe import GeoDataFrame
from spatialdata import SpatialData

from napari_sparrow.image._image import (
    _apply_transform,
    _get_boundary,
    _get_spatial_element,
    _unapply_transform,
)
from napari_sparrow.shape import intersect_rectangles
from napari_sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def plot_image(
    sdata: SpatialData,
    img_layer: str = "raw_image",
    channel: Optional[int | str | Iterable[int | str]] = None,
    z_slice: Optional[int] = None,
    crd: Optional[Tuple[int, int, int, int]] = None,
    output: Optional[str | Path] = None,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Plot an image based on given parameters.

    Parameters
    ----------
    sdata : SpatialData
        Data containing spatial information for plotting.
    img_layer : str, optional
        Image layer to be plotted. Default is "raw_image".
    channel : int or str or Iterable[int] or Iterable[str], optional
        Channel(s) to be displayed from the image.
    z_slice: int or None, optional
        The z_slice to visualize in case of 3D (c,z,y,x) image.
    crd : tuple of int, optional
        The coordinates for the region of interest in the format (xmin, xmax, ymin, ymax). If None, the entire image is considered, by default None.
    output : str or Path, optional
        Path to save the plot. If not provided, plot will be displayed.
    **kwargs : dict
        Additional arguments to be passed to the plot_shapes function.
    """
    plot_shapes(
        sdata,
        img_layer=img_layer,
        shapes_layer=None,
        channel=channel,
        z_slice=z_slice,
        crd=crd,
        output=output,
        **kwargs,
    )


def plot_labels(
    sdata: SpatialData,
    labels_layer: str = "segmentation_mask",
    z_slice: Optional[int] = None,
    crd: Optional[Tuple[int, int, int, int]] = None,
    output: Optional[str | Path] = None,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Plot a labels layer (masks) based on given parameters.

    Parameters
    ----------
    sdata : SpatialData
        Data containing spatial information for plotting.
    labels_layer : str, optional
        Labels layer to be plotted. Default is "segmentation_mask".
    z_slice: int or None, optional
        The z_slice to visualize in case of 3D (c,z,y,x) labels.
    crd : tuple of int, optional
        The coordinates for the region of interest in the format (xmin, xmax, ymin, ymax). If None, the entire image is considered, by default None.
    output : str or Path, optional
        Path to save the plot. If not provided, plot will be displayed.
    **kwargs : dict
        Additional arguments to be passed to the plot_shapes function.
    """
    plot_shapes(
        sdata,
        labels_layer=labels_layer,
        shapes_layer=None,
        z_slice=z_slice,
        crd=crd,
        output=output,
        **kwargs,
    )


def plot_shapes(
    sdata: SpatialData,
    img_layer: Optional[str | Iterable[str]] = None,
    labels_layer: Optional[str | Iterable[str]] = None,
    shapes_layer: Optional[str | Iterable[str]] = None,
    channel: Optional[int | str | Iterable[int] | Iterable[str]] = None,
    z_slice: Optional[int] = None,
    crd: Optional[Tuple[int, int, int, int]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    output: Optional[str | Path] = None,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Plot shapes and/or images/labels from a SpatialData object.
    The number of provided 'img_layer' or 'labels_layer' and 'shapes_layer' should be equal if both are iterables and if their length is greater than 1.

    Examples:

    1. For `img_layer=['raw_image', 'clahe']` and `shapes_layer=['segmentation_mask_boundaries', 'expanded_cells20']`:
    Subplots:
    - Column 1: 'raw_image' with 'segmentation_mask_boundaries'
    - Column 2: 'clahe' with 'expanded_cells20'

    2. For `img_layer=['raw_image', 'clahe']` and `shapes_layer='segmentation_mask_boundaries'`:
    Subplots:
    - Column 1: 'raw_image' with 'segmentation_mask_boundaries'
    - Column 2: 'clahe' with 'segmentation_mask_boundaries'

    3. For `img_layer=['raw_image', 'clahe']` and `shapes_layer=['segmentation_mask_boundaries']` (which behaves the same as the previous example):
    Subplots:
    - Column 1: 'raw_image' with 'segmentation_mask_boundaries'
    - Column 2: 'clahe' with 'segmentation_mask_boundaries'

    4. For `img_layer=['raw_image']` and `shapes_layer=['segmentation_mask_boundaries', 'expanded_cells20' ]`:
    Subplots:
    - Column 1: 'raw_image' with 'segmentation_mask_boundaries'
    - Column 2: 'raw_image' with 'expanded_cells20'

    5. For `img_layer=['raw_image', 'clahe']` and `shapes_layer=None`:
    Subplots:
    - Column 1: 'raw_image'
    - Column 2: 'clahe'

    When multiple channels are supplied as an Iterable, they will be displayed as rows in the image

    Parameters
    ----------
    sdata : SpatialData
        Data containing spatial information for plotting.
    img_layer : str or Iterable[str], optional
        Image layer(s) to be plotted. If not provided, and labels_layer is also not provided, the last added image layer is plotted.
        Displayed as columns in the plot, if multiple are provided.
    labels_layer : str or Iterable[str], optional
        Labels layer(s) to be plotted.
        Displayed as columns in the plot, if multiple are provided.
    shapes_layer : str or Iterable[str], optional
        Specifies which shapes to plot. If set to None, no shapes_layer is plotted.
        Displayed as columns in the plot, if multiple are provided.
    channel : int or str or Iterable[int] or Iterable[str], optional
        Channel(s) to be displayed from the image. Displayed as rows in the plot.
        If channel is None, get the number of channels from the first img_layer given as input.
        Ignored if img_layer is None and labels_layer is specified.
    z_slice: int or None, optional
        The z_slice to visualize in case of 3D (c,z,y,x) image/polygons.
    crd : tuple of int, optional
        The coordinates for the region of interest in the format (xmin, xmax, ymin, ymax). If None, the entire image is considered, by default None.
    figsize : Tuple[int, int], optional
        Size of the figure for plotting. If not provided, a default size is used based on the number of columns and rows.
    output : str or Path, optional
        Path to save the plot. If not provided, plot will be displayed.
    **kwargs : dict
        Additional arguments to be passed to the internal `_plot` function.

    Notes
    -----
    - This function offers advanced visualization options for `sdata` with support for multiple image layers, labels layers shape layers, and channels.
    - Either img_layer or labels_layer should be specified, not both.
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

    # Make code also work if user would provide another iterable than List
    layer = (
        list(layer)
        if isinstance(layer, Iterable) and not isinstance(layer, str)
        else [layer]
    )
    shapes_layer = (
        list(shapes_layer)
        if isinstance(shapes_layer, Iterable) and not isinstance(shapes_layer, str)
        else [shapes_layer]
    )
    if channel is not None:
        channel = (
            list(channel)
            if isinstance(channel, Iterable) and not isinstance(channel, str)
            else [channel]
        )

    # if multiple shapes are provided, and one img_layer, then len(shapes_layer) subfigures with same img_layer beneath are plotted.
    if len(layer) == 1 and shapes_layer != 1:
        layer = layer * len(shapes_layer)
    # if multiple img_layers are provided, and one shapes_layer, then len(img_layer) subfigures with same shapes_layer above are plotted.
    if len(shapes_layer) == 1 and layer != 1:
        shapes_layer = shapes_layer * len(layer)

    if (
        isinstance(layer, list)
        and isinstance(shapes_layer, list)
        and len(layer) != len(shapes_layer)
    ):
        raise ValueError(
            f"Length of '{layer}' is not equal to the length of shapes_layer '{shapes_layer}'."
        )

    nr_of_columns = max(len(layer), len(shapes_layer))

    if img_layer_type:
        # if channel is None, get the number of channels from the first img_layer given, maybe print a message about this.
        if channel is None:
            se = _get_spatial_element(sdata, layer=layer[0])
            channels = se.c.data
        else:
            channels = channel

    else:
        channels = [None]  # for labels_layer type, there are no channels

    nr_of_rows = len(channels)

    if figsize is None:
        figsize = (
            10 * nr_of_columns,
            10 * nr_of_rows,
        )
    fig, axes = plt.subplots(nr_of_rows, nr_of_columns, figsize=figsize)

    # Flattening axes to make iteration easier
    if nr_of_rows == 1 and nr_of_columns == 1:
        axes = np.array([axes])
    elif nr_of_rows == 1 or nr_of_columns == 1:
        axes = axes.reshape(-1)  # make it 1D if it isn't
    else:
        axes = axes.ravel()  # flatten the 2D array to 1D

    idx = 0
    for _channel in channels:
        for _layer, _shapes_layer in zip(layer, shapes_layer):
            _plot(
                sdata,
                axes[idx],
                img_layer=_layer if img_layer_type else None,
                labels_layer=_layer if not img_layer_type else None,
                shapes_layer=_shapes_layer,
                channel=_channel,
                z_slice=z_slice,
                crd=crd,
                **kwargs,
            )
            idx += 1

    plt.tight_layout()
    # Save the plot to output
    if output:
        fig.savefig(output)
    else:
        plt.show()
    plt.close()


def _plot(
    sdata: SpatialData,
    ax: plt.Axes,
    column: Optional[str] = None,
    cmap: str = "magma",
    img_layer: Optional[str] = None,
    labels_layer: Optional[str] = None,
    shapes_layer: Optional[str] = "segmentation_mask_boundaries",
    channel: Optional[int | str] = None,
    z_slice: Optional[int] = None,
    alpha: float = 0.5,
    crd: Tuple[int, int, int, int] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    shapes_layer_filtered: Optional[str | Iterable[str]] = None,
    img_title: bool = False,
    shapes_title: bool = False,
    channel_title: bool = True,
    aspect: str = "equal",
) -> plt.Axes:
    """
    Plots a SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        Data containing spatial information for plotting.
    ax : plt.Axes, optional
        Axes object to plot on.
    column : str or None, optional
        Column to base cell colors on. Can be an observation or variable column. If none provided, default color is used.
    cmap : str, default='magma'
        Colormap for the plot.
    img_layer : str or None, optional
        Image layer to be plotted. By default, the last added image layer is plotted.
    labels_layer : str or None, optional
        Labels layer to be plotted.
    shapes_layer : str or None, optional
        Specifies which shapes to plot. Default is 'segmentation_mask_boundaries'. If set to None, no shapes_layer is plot.
    channel : int or str or None, optional
        Channel to display from the image. If none provided, or if provided channel could not be found, first channel is plot.
        Ignored if img_layer is None and labels_layer is specified.
    z_slice: int or None, optional
        The z_slice to visualize in case of 3D (c,z,y,x) image/polygons.
    alpha : float, default=0.5
        Transparency level for the cells, given by the alpha parameter of matplotlib.
    crd : tuple of int, optional
        The coordinates for the region of interest in the format (xmin, xmax, ymin, ymax). If None, the entire image is considered, by default None.
    vmin : float or None, optional
        Lower bound for color scale for continuous data. Given as a percentile.
    vmax : float or None, optional
        Upper bound for color scale for continuous data. Given as a percentile.
    shapes_layer_filtered : str or Iterable[str], optional
        Extra shapes layers to plot. E.g. shapes filtered out in previous preprocessing steps.
    img_title: bool, default=False
        A flag indicating whether the image layer's name should be added to the title of the plot.
    shapes_title: bool, default=False
        A flag indicating whether the shapes layer's name should be added to the title of the plot.
    channel_title: bool, default=True
        A flag indicating whether the channel's name should be added to the title of the plot.
        Ignored if img_layer is None and labels_layer is specified.
    aspect : str, default='equal'
        Aspect ratio for the plot.

    Returns
    -------
    plt.Axes
        The axes with the plotted SpatialData.

    Notes
    -----
    - The function supports various visualization options such as image layers, shape layers, channels, color mapping, and custom regions.
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

    if shapes_layer_filtered is not None:
        shapes_layer_filtered = (
            list(shapes_layer_filtered)
            if isinstance(shapes_layer_filtered, Iterable)
            and not isinstance(shapes_layer_filtered, str)
            else [shapes_layer_filtered]
        )

    se = _get_spatial_element(sdata, layer=layer)

    # Update coords
    se, x_coords_orig, y_coords_orig = _apply_transform(se)

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
    size_im = (crd[1] - crd[0]) * (crd[3] - crd[2])
    if column is not None:
        if column + "_colors" in sdata.table.uns:
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "new_map",
                sdata.table.uns[column + "_colors"],
                N=len(sdata.table.uns[column + "_colors"]),
            )
        if column in sdata.table.obs.columns:
            column = sdata.table[
                sdata[shapes_layer].cx[crd[0] : crd[1], crd[2] : crd[3]].index, :
            ].obs[column]
        elif column in sdata.table.var.index:
            column = sdata.table[
                sdata[shapes_layer].cx[crd[0] : crd[1], crd[2] : crd[3]].index, :
            ].X[:, np.where(sdata.table.var.index == column)[0][0]]
        else:
            log.info(
                f"The column '{column}' is not a column in the dataframe sdata.table.obs, "
                "nor is it a gene name (sdata.table.var.index). The plot is made without taking into account this value."
            )
            column = None
            cmap = None
    else:
        cmap = None
    if vmin != None:
        vmin = np.percentile(column, vmin)
    if vmax != None:
        vmax = np.percentile(column, vmax)

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

    if shapes_layer is not None:
        polygons = sdata.shapes[shapes_layer].cx[crd[0] : crd[1], crd[2] : crd[3]]
        if z_slice is not None:
            polygons = _get_z_slice_polygons(polygons, z_slice=z_slice)
        if not polygons.empty:
            polygons.plot(
                ax=ax,
                edgecolor="white",
                column=column,
                linewidth=1 if size_im < 5000 * 10000 else 0,
                alpha=alpha,
                legend=True,
                aspect=1,
                cmap=cmap,
                vmax=vmax,  # np.percentile(column,vmax),
                vmin=vmin,  # np.percentile(column,vmin)
            )
        else:
            log.warning( f"Shapes layer {shapes_layer} was empty for crd {crd}." )
        if shapes_layer_filtered is not None:
            for i in shapes_layer_filtered:
                polygons = sdata.shapes[i].cx[crd[0] : crd[1], crd[2] : crd[3]]
                if z_slice is not None:
                    polygons = _get_z_slice_polygons(polygons, z_slice=z_slice)
                if not polygons.empty:
                    polygons.plot(
                        ax=ax,
                        edgecolor="red",
                        linewidth=1,
                        alpha=alpha,
                        legend=True,
                        aspect=1,
                        cmap="gray",
                    )
                else:
                    log.warning( f"Shapes layer {i} was empty for crd {crd}." )
    ax.axes.set_aspect(aspect)
    ax.set_xlim(crd[0], crd[1])
    ax.set_ylim(crd[2], crd[3])
    ax.invert_yaxis()
    titles = []
    if channel_title and img_layer_type:
        titles.append(f"{channel_name}={channel}")
    if img_title:
        titles.append(layer)
    if shapes_title and shapes_layer:
        titles.append(shapes_layer)
    title = ", ".join(titles)
    ax.set_title(title)
    # ax.axes.xaxis.set_visible(False)
    # ax.axes.yaxis.set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Restore coords
    se = _unapply_transform(se, x_coords_orig, y_coords_orig)

    return ax


def _get_z_slice_polygons(polygons: GeoDataFrame, z_slice: int) -> GeoDataFrame:
    def _get_z_slice(geometry: GeoSeries, z_value) -> bool:
        # return original geometry if geometry does not has z dimension
        if not geometry.has_z:
            return True

        if geometry.geom_type == "Polygon":
            for x, y, z in geometry.exterior.coords:
                if z == z_value:
                    return True

        elif geometry.geom_type == "MultiPolygon":
            for polygon in geometry.geoms:
                for x, y, z in polygon.exterior.coords:
                    if z == z_value:
                        return True

        return False

    return polygons[polygons["geometry"].apply(_get_z_slice, args=(z_slice,))]
