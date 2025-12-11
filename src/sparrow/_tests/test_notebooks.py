import importlib
import os

import nbformat
import pyrootutils
import pytest
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor


def run_notebook(notebook_path, timeout=600):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": os.path.dirname(notebook_path)}})
    except CellExecutionError as e:
        raise RuntimeError(f"Error executing the notebook '{notebook_path}': {e}") from e


@pytest.mark.skip
@pytest.mark.skipif(
    not importlib.util.find_spec("cellpose"),
    reason="requires the cellpose library",
)
@pytest.mark.parametrize(
    "notebook",
    [
        "SPArrOW_quickstart.ipynb",
    ],
)
def test_notebooks_harpy_transcriptomics(notebook):
    root = str(pyrootutils.setup_root(os.getcwd(), dotenv=True, pythonpath=True))

    run_notebook(os.path.join(root, "docs/tutorials/general", notebook))


@pytest.mark.skip
@pytest.mark.skipif(
    not importlib.util.find_spec("cellpose") or not importlib.util.find_spec("spatialdata_plot"),
    reason="requires the cellpose library",
)



@pytest.mark.skip
@pytest.mark.skipif(
    not importlib.util.find_spec("textalloc") or not importlib.util.find_spec("spatialdata_plot"),
    reason="requires the textalloc library",
)



@pytest.mark.skip
@pytest.mark.skipif(
    not importlib.util.find_spec("textalloc")
    or not importlib.util.find_spec("joypy")
    or not importlib.util.find_spec("spatialdata_plot"),
    reason="requires the textalloc library",
)


@pytest.mark.skip
@pytest.mark.parametrize(
    "notebook",
    [
        "coordinate_systems.ipynb",
    ],
)
def test_notebooks_coordinate_systems(notebook):
    root = str(pyrootutils.setup_root(os.getcwd(), dotenv=True, pythonpath=True))

    run_notebook(os.path.join(root, "docs/tutorials/advanced", notebook))

@pytest.mark.skip
@pytest.mark.skipif(
    not importlib.util.find_spec("rasterio"),
    reason="requires the rasterio library",
)
@pytest.mark.parametrize(
    "notebook",
    [
        "Rasterize_and_vectorize.ipynb",
    ],
)
def test_notebooks_rasterize_vectorize(notebook):
    root = str(pyrootutils.setup_root(os.getcwd(), dotenv=True, pythonpath=True))

    run_notebook(os.path.join(root, "docs/tutorials/advanced", notebook))


@pytest.mark.skip
@pytest.mark.parametrize(
    "notebook",
    [
        "Harpy_aggregate_rasters.ipynb",
    ],
)
def test_notebooks_aggregate_rasters(notebook):
    root = str(pyrootutils.setup_root(os.getcwd(), dotenv=True, pythonpath=True))

    run_notebook(os.path.join(root, "docs/tutorials/advanced", notebook))


@pytest.mark.skip
@pytest.mark.skipif(
    not importlib.util.find_spec("cellpose") or not importlib.util.find_spec("basicpy"),
    reason="requires the cellpose and basicpy libraries",
)
@pytest.mark.parametrize(
    "notebook",
    [
        "Harpy_how_to_start.ipynb",
    ],
)
def test_notebook_harpy_pipeline(notebook):
    root = str(pyrootutils.setup_root(os.getcwd(), dotenv=True, pythonpath=True))

    run_notebook(os.path.join(root, "docs/tutorials/general", notebook))
