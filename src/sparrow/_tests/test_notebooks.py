from __future__ import annotations

import importlib.util
from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor


# Resolve the repository root from this test module so notebook paths are independent of cwd.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

# Resolve notebook paths from the tutorials submodule's stable top-level directory.
TUTORIALS_ROOT = REPOSITORY_ROOT / "docs" / "tutorials"

# Keep the notebook index in one place so validation covers every tutorial notebook.
GENERAL_NOTEBOOKS = (
    "CosMx_tutorial.ipynb",
    "Merscope_tutorial.ipynb",
    "SPArrOW_how_to_start.ipynb",
    "VisiumHD_tutorial.ipynb",
)
ADVANCED_NOTEBOOKS = (
    "coordinate_systems.ipynb",
    "retrain_cellpose_sparrow_tutorial.ipynb",
    "xenium_multistaining_tutorial.ipynb",
)

# Build notebooks paths from the canonical content root for the lightweight integrity check.
INDEXED_NOTEBOOKS = tuple(
    TUTORIALS_ROOT / category / notebook
    for category, notebooks in (
        ("general", GENERAL_NOTEBOOKS),
        ("advanced", ADVANCED_NOTEBOOKS),
    )
    for notebook in notebooks
)


def run_notebook(notebook_path: Path, timeout: int = 600) -> None:
    # Read the notebook as nbformat 4 before handing it to nbconvert.
    with notebook_path.open(encoding="utf-8") as notebook_file:
        notebook = nbformat.read(notebook_file, as_version=4)

    # Execute from the notebook directory so relative paths behave as they do interactively.
    executor = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    try:
        executor.preprocess(notebook, {"metadata": {"path": str(notebook_path.parent)}})
    except CellExecutionError as e:
        raise RuntimeError(f"Error executing the notebook '{notebook_path}': {e}") from e

# Test that all indexed notebooks exist and parse correctly.
@pytest.mark.parametrize("notebook_path", INDEXED_NOTEBOOKS)
def test_indexed_notebooks_exist_and_parse(notebook_path: Path) -> None:
    # Fail quickly with a clear message when the submodule is incomplete or a notebook is malformed.
    assert notebook_path.is_file(), f"Indexed notebook does not exist: {notebook_path}"
    with notebook_path.open(encoding="utf-8") as notebook_file:
        nbformat.read(notebook_file, as_version=4)


@pytest.mark.parametrize("notebook", ["coordinate_systems.ipynb"])
def test_notebooks_coordinate_systems(notebook: str) -> None:
    # Execute the advanced coordinate-system workflow from the tutorials submodule.
    run_notebook(TUTORIALS_ROOT / "advanced" / notebook)


@pytest.mark.skipif(
    not importlib.util.find_spec("cellpose")
    or not importlib.util.find_spec("basicpy")
    or not importlib.util.find_spec("squidpy"),
    reason="requires the cellpose, basicpy, and squidpy libraries",
)
@pytest.mark.parametrize("notebook", ["SPArrOW_how_to_start.ipynb"])
def test_notebook_sparrow_pipeline(notebook: str) -> None:
    # Execute the general pipeline workflow from the tutorials submodule.
    run_notebook(TUTORIALS_ROOT / "general" / notebook)
