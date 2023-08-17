from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Iterable
import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from spatialdata import SpatialData

from napari_sparrow.shape import intersect_rectangles
from napari_sparrow.image._image import (
    _apply_transform,
    _unapply_transform,
    _get_image_boundary,
)


def plot_image(
    sdata: SpatialData,
    img_layer: str = "raw_image",
    channel: Optional[int | Iterable[int]] = None,
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
    channel : int or Iterable[int], optional
        Channel(s) to be displayed from the image.
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
        crd=crd,
        output=output,
        **kwargs,
    )


def plot_shapes(
    sdata: SpatialData,
    img_layer: str | Iterable[str] = None,
    shapes_layer: str | Iterable[str] = None,
    channel: Optional[int | Iterable[int]] = None,
    crd: Optional[Tuple[int, int, int, int]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    output: Optional[str | Path] = None,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Plot shapes and/or images from a SpatialData object.
    The number of provided 'img_layer' and 'shapes_layer' should be equal if both are iterables and if their length is greater than 1.

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
        Image layer(s) to be plotted. If not provided, the last added image layer is plotted.
        Displayed as columns in the plot, if multiple are provided.
    shapes_layer : str or Iterable[str], optional
        Specifies which shapes to plot. If set to None, no shapes_layer is plotted.
        Displayed as columns in the plot, if multiple are provided.
    channel : int or Iterable[int], optional
        Channel(s) to be displayed from the image. Displayed as rows in the plot.
        If channel is None, get the number of channels from the first img_layer given as input.
    crd : tuple of int, optional
        The coordinates for the region of interest in the format (xmin, xmax, ymin, ymax). If None, the entire image is considered, by default None.
    figsize : Tuple[int, int], optional
        Size of the figure for plotting. If not provided, a default size is used based on the number of columns and rows.
    output : str or Path, optional
        Path to save the plot. If not provided, plot will be displayed.
    **kwargs : dict
        Additional arguments to be passed to the internal _plot_shapes function.

    Notes
    -----
    - This function offers advanced visualization options for spatial data with support for multiple image layers, shape layers, and channels.
    """
    # need this to be able to determine the number of channels if channels would be None
    if img_layer is None:
        img_layer = [*sdata.images][-1]

    # Make code also work if user would provide another iterable than List
    img_layer = (
        list(img_layer)
        if isinstance(img_layer, Iterable) and not isinstance(img_layer, str)
        else [img_layer]
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
    if len(img_layer) == 1 and shapes_layer != 1:
        img_layer = img_layer * len(shapes_layer)
    # if multiple img_layers are provided, and one shapes_layer, then len(img_layer) subfigures with same shapes_layer above are plotted.
    if len(shapes_layer) == 1 and img_layer != 1:
        shapes_layer = shapes_layer * len(img_layer)

    if (
        isinstance(img_layer, list)
        and isinstance(shapes_layer, list)
        and len(img_layer) != len(shapes_layer)
    ):
        raise ValueError(
            f"Length of img_layer '{img_layer}' is not equal to the length of shapes_layer '{shapes_layer}'."
        )

    nr_of_columns = max(len(img_layer), len(shapes_layer))

    # if channel is None, get the number of channels from the first img_layer given, maybe print a message about this.
    if channel is None:
        channels = sdata[img_layer[0]].c.data
    else:
        channels = channel

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
        for _img_layer, _shapes_layer in zip(img_layer, shapes_layer):
            _plot_shapes(
                sdata,
                axes[idx],
                img_layer=_img_layer,
                shapes_layer=_shapes_layer,
                channel=_channel,
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


def _plot_shapes(  # FIXME: rename, this does not always plot a shapes layer anymore
    sdata: SpatialData,
    ax: plt.Axes,
    column: Optional[str] = None,
    cmap: str = "magma",
    img_layer: Optional[str] = None,
    shapes_layer: Optional[str] = "segmentation_mask_boundaries",
    channel: Optional[int] = None,
    alpha: float = 0.5,
    crd: Tuple[int, int, int, int] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    plot_filtered: bool = False,
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
    shapes_layer : str or None, optional
        Specifies which shapes to plot. Default is 'segmentation_mask_boundaries'. If set to None, no shapes_layer is plot.
    channel : int or None, optional
        Channel to display from the image. If none provided, or if provided channel could not be found, first channel is plot.
    alpha : float, default=0.5
        Transparency level for the cells, given by the alpha parameter of matplotlib.
    crd : tuple of int, optional
        The coordinates for the region of interest in the format (xmin, xmax, ymin, ymax). If None, the entire image is considered, by default None.
    vmin : float or None, optional
        Lower bound for color scale for continuous data. Given as a percentile.
    vmax : float or None, optional
        Upper bound for color scale for continuous data. Given as a percentile.
    plot_filtered : bool, default=False
        Whether to plot the cells that were filtered out in previous steps.
    img_title: bool, default=False
        A flag indicating whether the image layer's name should be added to the title of the plot.
    shapes_title: bool, default=False
        A flag indicating whether the shapes layer's name should be added to the title of the plot.
    channel_title: bool, default=True
        A flag indicating whether the channel's name should be added to the title of the plot.
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
    if img_layer is None:
        img_layer = [*sdata.images][-1]

    si = sdata.images[img_layer]

    # Update coords
    si, x_coords_orig, y_coords_orig = _apply_transform(si)

    image_boundary = _get_image_boundary(si)

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
            print(
                "The column defined in the function isnt a column in obs, nor is it a gene name, the plot is made without taking into account this value."
            )
            column = None
            cmap = None
    else:
        cmap = None
    if vmin != None:
        vmin = np.percentile(column, vmin)
    if vmax != None:
        vmax = np.percentile(column, vmax)

    if channel is None:
        # if channel is None, plot the first channel
        channel = si.c.data[0]
        # if channel not in spatialimage object, plot the first channel
    elif channel not in si.c.data:
        _channel = channel
        channel = si.c.data[0]
        warnings.warn(
            (
                f"Provided channel '{_channel}' not in list of available channels '{si.c.data}'"
                f"for provided img_layer '{img_layer}'. Falling back to plotting first available channel '{channel}' for this img_layer."
            )
        )

    channel_name = si.c.name

    si.isel(c=channel).squeeze().sel(
        x=slice(crd[0], crd[1]), y=slice(crd[2], crd[3])
    ).plot.imshow(cmap="gray", robust=True, ax=ax, add_colorbar=False)

    if shapes_layer:
        sdata[shapes_layer].cx[crd[0] : crd[1], crd[2] : crd[3]].plot(
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
        if plot_filtered:
            for i in [*sdata.shapes]:
                if "filtered" in i:
                    sdata[i].cx[crd[0] : crd[1], crd[2] : crd[3]].plot(
                        ax=ax,
                        edgecolor="red",
                        linewidth=1,
                        alpha=alpha,
                        legend=True,
                        aspect=1,
                        cmap="gray",
                    )
    ax.axes.set_aspect(aspect)
    ax.set_xlim(crd[0], crd[1])
    ax.set_ylim(crd[2], crd[3])
    ax.invert_yaxis()
    titles = []
    if channel_title:
        titles.append(f"{channel_name}={channel}")
    if img_title:
        titles.append(img_layer)
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
    si = _unapply_transform(si, x_coords_orig, y_coords_orig)

    return ax
