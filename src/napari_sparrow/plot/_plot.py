from pathlib import Path
from typing import List, Optional, Tuple
import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from spatialdata import SpatialData

from napari_sparrow.shape import intersect_rectangles
from napari_sparrow.image._image import _apply_transform, _unapply_transform, _get_image_boundary


def plot_image(  # TODO: perhaps plot_image and plot_shapes can be merged into 1 function? Of if we keep this function, see if we need to rename it a bit or change its API.
    sdata: SpatialData,
    output_path: Optional[str | Path] = None,
    crd: Optional[List[int]] = None,
    layer: str = "image",
    channel: Optional[int] = None,
    aspect: str = "equal",
    figsize: Tuple[int, int] = (10, 10),
):
    plot_shapes(
        sdata,
        output=output_path,
        crd=crd,
        img_layer=layer,
        shapes_layer=None,
        channel=channel,
        aspect=aspect,
        figsize=figsize,
    )


def plot_shapes(  # FIXME: rename, this does not always plot a shapes layer anymore
    sdata,
    column: Optional[str] = None,
    cmap: str = "magma",
    img_layer: Optional[str] = None,
    channel: Optional[int] = None,
    shapes_layer: Optional[str] = "segmentation_mask_boundaries",
    alpha: float = 0.5,
    crd=None,
    output: Optional[str | Path] = None,
    vmin=None,
    vmax=None,
    plot_filtered=False,
    aspect: str = "equal",
    figsize=(20, 20),
) -> None:
    """
    This function plots an sdata object, with the cells on top. On default it plots the image layer that was added last.
    The default color is blue if no color is given as input.
    Column: determines based on which column the cells need to be colored. Can be an obs column or a var column.
    img_layer: the image layer that needs to be plotted, the last on on default
    shapes_layer: which shapes to plot, the default is nucleus_boundaries, but when performing an expansion it can be another layer.
    alpha: the alpha-parameter of matplotlib: transperancy of the cells
    crd: the crop that needs to be plotted, if none is given, the whole region is plotted, list of four coordinates
    output: whether you want to save it as an output or not, default is none and then plot is shown.
    vmin/vmax: adapting the color scale for continous data: give the percentile for which to color min and max.
    ax: whne wanting to add the plot to another plot
    plot_filtered: whether or not to plot the cells that were filtered out during previous steps, this is a control function.
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

    channels = [channel] if channel is not None else si.c.data

    for ch in channels:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        si.isel(c=ch).squeeze().sel(
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
        ax.set_title("")
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # Save the plot to ouput
        if output:
            fig.savefig(f"{output}_{ch}")
        else:
            plt.show()
        plt.close()

    # Restore coords
    si = _unapply_transform(si, x_coords_orig, y_coords_orig)