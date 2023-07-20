from typing import Optional
import warnings

import numpy as np
import matplotlib.pyplot as plt

from napari_sparrow.image._image import (_get_image_boundary, _apply_transform, _unapply_transform)
from napari_sparrow.shape import intersect_rectangles


def segment(
    sdata,
    crd=None,
    layer: Optional[str] = None,
    channel: Optional[int] = None,
    shapes_layer="segmentation_mask_boundaries",
    output: Optional[str] = None,
) -> None:
    if layer is None:
        layer = [*sdata.images][-1]

    si = sdata.images[layer]

    # Note: sdata[shapes_layer] stores the segmentation outlines in global coordinates, whereas
    # the SpatialImage sdata.images['clahe'] has a transformation associated with it which handles the position
    # of a possible crop rectangle. However, in the code below will use xarray.plot.imshow() to plot this image
    # together with the outlines in the same matplotlib plot. This requires us to transform the image to the same
    # coordinate system as the outlines on the plot. The straightforward way to do so is via the 'extent' parameter
    # for imshow, but it turns out that xarray.plot.dataarray_plot.py's imshow() simply ignores the 'extent' argument
    # that it receives, and calculates it own extent from the SpatialImage's x and y coords array. That is why we
    # temporarily overwrite the x and y coords in the SpatialImage with a translated version before plotting, and then
    # restore it afterwards.

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

    channels = [channel] if channel is not None else si.c.data

    for ch in channels:
        fig, ax = plt.subplots(1, 2, figsize=(20, 20))

        # Contrast enhanced image
        si.isel(c=ch).squeeze().sel(
            x=slice(crd[0], crd[1]), y=slice(crd[2], crd[3])
        ).plot.imshow(cmap="gray", robust=True, ax=ax[0], add_colorbar=False)

        # Contrast enhanced image with segmentation shapes overlaid on top of it
        si.isel(c=ch).squeeze().sel(
            x=slice(crd[0], crd[1]), y=slice(crd[2], crd[3])
        ).plot.imshow(cmap="gray", robust=True, ax=ax[1], add_colorbar=False)
        sdata[shapes_layer].cx[crd[0] : crd[1], crd[2] : crd[3]].plot(
            ax=ax[1],
            edgecolor="white",
            linewidth=1,
            alpha=0.5,
            legend=True,
            aspect=1,
        )
        for i in range(len(ax)):
            ax[i].axes.set_aspect("equal")
            ax[i].set_xlim(crd[0], crd[1])
            ax[i].set_ylim(crd[2], crd[3])
            ax[i].invert_yaxis()
            ax[i].set_title("")
            # ax.axes.xaxis.set_visible(False)
            # ax.axes.yaxis.set_visible(False)
            ax[i].spines["top"].set_visible(False)
            ax[i].spines["right"].set_visible(False)
            ax[i].spines["bottom"].set_visible(False)
            ax[i].spines["left"].set_visible(False)

        # Save the plot to output
        if output:
            plt.close(fig)
            fig.savefig(f"{output}_{ch}")

    # Restore coords
    si = _unapply_transform(si, x_coords_orig, y_coords_orig)
