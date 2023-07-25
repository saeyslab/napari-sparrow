from typing import Union, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import geopandas
import spatialdata
from dask.dataframe.core import DataFrame as DaskDataFrame

from napari_sparrow.image._image import _get_translation, _apply_transform, _unapply_transform
from napari_sparrow.shape._shape import _extract_boundaries_from_geometry_collection


def sanity_plot_transcripts_matrix(
    xarray: Union[np.ndarray, xr.DataArray],
    in_df: Optional[Union[pd.DataFrame, DaskDataFrame]] = None,
    polygons: Optional[geopandas.GeoDataFrame] = None,
    plot_cell_number: bool = False,
    n: Optional[int] = None,
    name_x: str = "x",
    name_y: str = "y",
    name_gene_column: str = "gene",
    gene: Optional[str] = None,
    crd: Optional[Tuple[int, int, int, int]] = None,
    cmap: str = "gray",
    output: Optional[Union[Path, str]] = None,
):
    # in_df can be dask dataframe or pandas dataframe

    # plot for sanity check

    fig, ax = plt.subplots(figsize=(10, 10))

    if isinstance(xarray, np.ndarray):
        # CHECKME: is this case useful? Will the input always have a channel dimension?
        xarray = spatialdata.models.Image2DModel.parse(xarray, dims=("c", "y", "x"))

    tx, ty = _get_translation(xarray)

    if crd is None:
        crd = [
            tx,
            tx + xarray.sizes["x"],
            ty,
            ty + xarray.sizes["y"],
        ]

    xarray, x_coords_orig, y_coords_orig = _apply_transform(xarray)

    xarray.squeeze().sel(x=slice(crd[0], crd[1]), y=slice(crd[2], crd[3])).plot.imshow(
        cmap=cmap, robust=True, ax=ax, add_colorbar=False
    )

    xarray = _unapply_transform(xarray, x_coords_orig, y_coords_orig)

    # update so that a sample is taken from the dataframe (otherwise plotting takes too long), i.e. take n points max

    if in_df is not None:
        # query first and then slicing gene is faster than vice versa
        in_df = in_df.query(
            f"{crd[0]} <= {name_x} <= {crd[1]} and {crd[2]} <= {name_y} <= {crd[3]}"
        )

        if gene:
            in_df = in_df[in_df[name_gene_column] == gene]

        # we do not sample a fraction of the transcripts if a specific gene is given
        else:
            size = len(in_df)

            print(f"size before sampling is {size}")

            if n is not None and size > n:
                fraction = n / size
                in_df = in_df.sample(frac=fraction)

        if isinstance(in_df, DaskDataFrame):
            in_df = in_df.compute()

        print(f"Plotting {in_df.shape[0]} transcripts.")

        if gene:
            alpha = 0.5
        else:
            alpha = 0.2

        ax.scatter(in_df[name_x], in_df[name_y], color="r", s=8, alpha=alpha)

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
