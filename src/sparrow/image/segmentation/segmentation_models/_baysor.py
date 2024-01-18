import json
import os
import subprocess
import tempfile
from pathlib import Path

import geopandas as gpd
import rasterio
import shapely
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from pandas import DataFrame as PandasDataFrame
from PIL import Image
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, shape
from shapely.validation import make_valid

from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def _baysor(
    img: NDArray,
    df: PandasDataFrame,
    name_x: str,
    name_y: str,
    name_gene: str,
    config_path: str | Path,
    output_dir: str | Path,
    threads: int,
    diameter: int = 40,  # this is scale in baysor, should be approx equal to the expected cell radius
    use_prior_segmentation: bool = True,
    prior_confidence: int = 0.2,  # expected quality of the prior (i.e. masks). 0.0 will make algorithm ignore the prior, while 1.0 restricts the algorithm from contradicting the prior.
) -> NDArray:
    # img is (z,y,x,c), and returned masks are also (z,y,x,c)
    log.info(f"Prior segmentation: {use_prior_segmentation}")
    log.info(f"Prior confidence: {prior_confidence}")
    assert (
        img.ndim == 4
    ), "Please provide img with (z,y,x,c) dimension. z and c dimension should be 1."
    assert (
        img.shape[0] == 1
    ), "Currently only 2D segmentation is supported. I.e. z-dimension should be 1"
    assert img.shape[-1] == 1, "Channel dimension should be 1."
    assert name_x in df.columns, f"DataFrame should contain 'x' coordinate '{name_x}'."
    assert name_y in df.columns, f"DataFrame should contain 'y' coordinate '{name_y}'."
    assert (
        name_gene in df.columns
    ), f"DataFrame should contain 'gene' column. '{name_gene}'."

    # squeeze the trivial c-channel
    img = img.squeeze(-1)
    # currently only support 2D segmentation with baysor.
    img = img.squeeze(0)

    x_min = df.x.min()
    x_max = df.x.max()

    y_min = df.y.min()
    y_max = df.y.max()

    y_max

    with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir:
        df.to_csv(os.path.join(temp_dir, "transcripts.csv"))
        # Save the image as a TIFF file
        if use_prior_segmentation:
            image = Image.fromarray(img)
            masks_path = os.path.join(temp_dir, "masks.tiff")
            image.save(masks_path, format="TIFF")
        else:
            masks_path = ""

        output_html = os.path.join(temp_dir, "output.html")

        # command to run baysor
        # TODO add force 2D here
        command = (
            f"JULIA_NUM_THREADS={threads} baysor run "
            f"-s {diameter} "
            f"-x {name_x} "
            f"-y {name_y} "
            f"-g {name_gene} "
            f"-c {config_path} "
            f"-o {output_html} "
            f"--prior-segmentation-confidence={prior_confidence} "
            f"{os.path.join( temp_dir, 'transcripts.csv' ) } "
            "--save-polygons=geojson "
            f"{masks_path}"
        )

        # command = "JULIA_NUM_THREADS=8 baysor run -s 40 -c config.toml -o output_python.html 20272_slide1_A1-1_results_4288_2144_baysor.txt --save-polygons=geojson #[PRIOR_SEGMENTATION]"

        # Run the command and capture output
        process = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Get the standard output
        output = process.stdout

        # Get the standard error
        error = process.stderr

        log.info(output)
        log.error(error)

        polygons = _read_baysor(
            path_polygons=os.path.join(temp_dir, "output_polygons.json")
        )

    # convert polygons to masks
    masks = rasterio.features.rasterize(
        zip(
            polygons.geometry,
            polygons.index.values.astype(float),
        ),
        out_shape=[img.shape[0], img.shape[1]],
        dtype="uint32",
        fill=0,
    )

    # add z and c dimension to masks
    masks = masks[None, ..., None]

    return masks


def _read_baysor(path_polygons: str | Path, min_vertices: int = 4) -> GeoDataFrame:
    with open(path_polygons) as f:
        polygons_dict = json.load(f)
        polygons_dict = {c["cell"]: c for c in polygons_dict["geometries"]}

    polygons_dict = {
        num: data
        for num, data in polygons_dict.items()
        if len(data["coordinates"][0]) >= min_vertices
    }

    polygons = [shape(data) for _, data in polygons_dict.items()]

    gdf = gpd.GeoDataFrame(geometry=polygons)

    gdf.geometry = gdf.geometry.map(lambda cell: _ensure_polygon(make_valid(cell)))
    gdf = gdf[~gdf.geometry.isna()]

    return gdf


# taken from sopa https://github.com/gustaveroussy/sopa
def _ensure_polygon(cell: Polygon | MultiPolygon | GeometryCollection) -> Polygon:
    """Ensures that the provided cell becomes a Polygon

    Args:
        cell: A shapely Polygon or MultiPolygon or GeometryCollection

    Returns:
        The shape as a Polygon
    """
    cell = shapely.make_valid(cell)

    if isinstance(cell, Polygon):
        if cell.interiors:
            cell = Polygon(list(cell.exterior.coords))
        return cell

    if isinstance(cell, MultiPolygon):
        return max(cell.geoms, key=lambda polygon: polygon.area)

    if isinstance(cell, GeometryCollection):
        geoms = [geom for geom in cell.geoms if isinstance(geom, Polygon)]

        if not geoms:
            log.info(
                f"Removing cell of type {type(cell)} as it contains no Polygon geometry"
            )
            return None

        return max(geoms, key=lambda polygon: polygon.area)

    log.info(f"Removing cell of unknown type {type(cell)}")
    return None
