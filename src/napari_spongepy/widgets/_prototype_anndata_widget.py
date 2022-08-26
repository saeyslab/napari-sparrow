import math
import pathlib

import napari
import napari.layers
import napari.types
import numpy as np
import pandas as pd
from magicgui import magic_factory

RESOLVE_PIXEL_SIZE = 0.138  # CHECKME: is this indeed the pixel size of the Resolve microscope images, in microns?


COLOR_CYCLE0 = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])

COLOR_CYCLE1 = np.array(
    [[1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 0, 1], [0, 1, 1, 1], [0, 0, 1, 1], [1, 0, 0, 1]]
)

COLOR_CYCLE2 = (
    np.array(
        [
            [255, 225, 25],
            [255, 80, 70],
            [75, 195, 90],
            [245, 170, 65],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
            [250, 190, 190],
            [230, 190, 255],
            [255, 250, 200],
            [170, 255, 195],
            [255, 215, 180],
        ]
    )
    / 255
)


@magic_factory(
    call_button="Visualize", density_radius_micrometer={"min": 0.1, "max": 10.0}
)
def anndata_widget(
    viewer: napari.Viewer,
    image: napari.layers.Image,
    transcripts_file: pathlib.Path = pathlib.Path(""),
    genes_to_plot: str = "",  # comma-separated list of genes to plot, e.g. 'Acta2,Xcr1'; gene names are case sensitive
    plot_density: bool = True,
    density_radius_micrometer: float = 3.45,  # 3.45 is Polylux default
    plot_points: bool = True,
    point_size: int = 10,
):
    print("About to Visualize transcripts")

    # Read CSV file with transcript coordinates and name.
    # The resulting dataframe df has a row for each transcript detected in the Resolve experiment;
    # each row is "x y z gene" with the (x,y,z) coordinate of the position where the transcript for 'gene' was detected.
    df = pd.read_csv(
        transcripts_file,
        delimiter="\t",
        header=None,
        usecols=[0, 1, 2, 3],
        names=["x", "y", "z", "gene"],
    )

    # Assign a unique integer value with each unique gene.
    # This is useful for coloring identical genes with the the same color lateron.
    gene_codes, unique_genes = pd.factorize(df["gene"], sort=True)
    df["gene_index"] = gene_codes

    print(
        f"Got {len(gene_codes)} transcripts for these {len(unique_genes)} unique genes: {list(unique_genes)}"
    )

    # Make a list with the genes to display.
    # Remove whitespace and split comma-separated string with gene names, e.g. 'Acta2, Xcr1' -> ['Acta2', 'Xcr1']
    selected_genes = genes_to_plot.replace(" ", "").split(",")
    print(f"Selected genes={selected_genes}")

    # Slice the transcripts dataframe to only hold transcripts for the selected genes.
    selected_df = df[df["gene"].isin(selected_genes)]

    if selected_df.empty:
        print(
            "No transcripts for the specified list of genes. Is the list empty, or does it have typos?"
        )
        return

    if plot_density:
        layer = _make_transcript_density_layer(
            image, selected_df, density_radius_micrometer
        )
        viewer.add_image(
            layer[0], **layer[1]
        )  # FIXME: we want to layer to be updated if it already exists

    if plot_points:
        layer = _make_transcript_points_layer(selected_df, point_size)
        viewer.add_points(
            layer[0], **layer[1]
        )  # FIXME: we want to layer to be updated if it already exists


def _make_transcript_density_layer(
    image: napari.layers.Image, df: pd.DataFrame, density_radius_micrometer: float
) -> napari.types.LayerDataTuple:

    print("Calculating density map")
    density_radius_pixels = math.ceil(density_radius_micrometer / RESOLVE_PIXEL_SIZE)
    spots = df[["y", "x"]].to_numpy()
    density = _calculate_density_map(image.data.shape, spots, density_radius_pixels)

    max_density = np.max(density)
    if max_density > 0:
        density_image = density / np.max(density)
    else:
        density_image = np.zeros(image.data.shape, dtype=np.uint8)

    layer = (
        density_image,
        {"name": "transcript density", "colormap": "inferno", "opacity": 0.75},
    )
    return layer


def _make_transcript_points_layer(
    df: pd.DataFrame, point_size: int
) -> napari.types.LayerDataTuple:

    point_properties = {
        "gene": list(
            df["gene"]
        ),  # gene name, shown when user hovers over a point in the GUI
        "gene_index": list(
            df["gene_index"]
        ),  # integer gene index, determines the point's color
    }

    layer = (
        df[["y", "x"]],  # point coordinates
        {
            "name": "transcripts",
            "size": point_size,
            "properties": point_properties,
            "face_color": "gene_index",
            "face_color_cycle": COLOR_CYCLE2,
            "edge_width": 0,
        },
    )
    return layer


# Note: we could perhaps instead use scikit-learn's KernelDensity(kernel="linear", ...)
# to calculate the density map.
# https://scikit-learn.org/stable/modules/density.html#kernel-density-estimation


def _calculate_density_map(map_shape, spots: np.ndarray, radius: int):
    # radius is in pixels; spots is a 2D array where each row holds the integer (x,y) coordinates of a spot

    # Calculate the kernel, a 2D matrix with the "weight" of each pixel in a spot with given radius.
    kernel = _calculate_kernel(radius)

    # Initialize density map to zero, but pad it so that we can still place the
    # kernel matrix over spots near the edge of the imaged area.
    density_map = np.pad(np.zeros(map_shape, dtype=float), radius)

    # Accumulate the spot density for all spots.
    # (= Add the kernel matrix to a slice of the density map
    # of the same size and at the correct position.)
    for spot in spots:
        i, j = spot
        density_map[i : i + kernel.shape[0], j : j + kernel.shape[1]] += kernel

    # Return the density map, but with the padding removed again.
    return density_map[radius:-radius, radius:-radius]


def _pixelsInCircle(radius: int) -> int:
    # Assuming pixels of size 1 unit, and a circle of given radius
    # centered on the middle of a square pixel, this function returns
    # the number of pixels whose center is strictly inside that circle.
    numInRadius = 0
    for i in range(-radius - 1, radius + 1):
        for j in range(-radius - 1, radius + 1):
            if i * i + j * j < radius * radius:
                numInRadius += 1
    return numInRadius


assert [_pixelsInCircle(r) for r in range(10)] == [
    0,
    1,
    9,
    25,
    45,
    69,
    109,
    145,
    193,
    249,
], "Assertion failed for _pixelsInRadius()"


def _calculate_kernel(radius: int):
    numPixelsInCircle: int = _pixelsInCircle(radius)
    pixel_area: float = RESOLVE_PIXEL_SIZE * RESOLVE_PIXEL_SIZE
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=float)
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            distance = math.sqrt(i * i + j * j)
            if distance < radius:
                kernel[i + radius, j + radius] = (
                    ((radius - distance) / radius) * pixel_area / numPixelsInCircle
                )
    return kernel
