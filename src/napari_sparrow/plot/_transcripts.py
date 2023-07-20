from typing import Optional
import matplotlib.pyplot as plt

from napari_sparrow.image._image import (_get_translation, _apply_transform, _unapply_transform)


def transcript_density(blurred, sdata, layer: Optional[str] = None, crd=None):  # TODO: add type hints
    if layer is None:
        layer = [*sdata.images][-1]  # typically layer will be the "clahe" layer

    # TODO: find intersection of the translation + size of 'layer' with the (optional) 'crd' crop rectangle

    si, x_coords_orig, y_coord_orig = _apply_transform(sdata[layer])

    fig, ax = plt.subplots(1, 2, figsize=(20, 20))
    if crd:
        ax[0].imshow(
            blurred[crd[2] : crd[3], crd[0] : crd[1]],
            cmap="magma",
            vmax=5,
            extent=[crd[0], crd[1], crd[3], crd[2]],
        )
        print(_get_translation(si))
        si.squeeze().sel(x=slice(crd[0], crd[1]), y=slice(crd[2], crd[3])).plot.imshow(
            cmap="gray", robust=True, ax=ax[1], add_colorbar=False
        )
    else:
        ax[0].imshow(blurred, cmap="magma", vmax=5)
        print(_get_translation(si))
        si.squeeze().plot.imshow(cmap="gray", robust=True, ax=ax[1], add_colorbar=False)

    ax[1].axes.set_aspect("equal")
    ax[1].invert_yaxis()

    ax[0].set_title("Transcript density")
    ax[1].set_title("Corrected image")

    si = _unapply_transform(si, x_coords_orig, y_coord_orig)


