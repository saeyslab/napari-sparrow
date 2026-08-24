from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr
from dask.array import Array
from spatialdata.transformations import Identity, get_transformation
from xarray import DataTree

from sparrow.io._cosmx import (
    _discover_files,
    _load_keep_gene_names,
    cosmx,
)
from sparrow.table._allocation import allocate
from sparrow.utils._keys import _GENES_KEY, _INSTANCE_KEY, _REGION_KEY


def _write_cosmx_dataset(root: Path) -> Path:
    """Write a minimal modern CosMx export that the native reader can load."""
    root.mkdir(parents=True, exist_ok=True)
    images_dir = root / "CellComposite"
    labels_dir = root / "CellLabels"
    images_dir.mkdir()
    labels_dir.mkdir()

    # Write grouped image channels and a matching global integer label pyramid.
    _write_zarr_stores(images_dir, labels_dir)

    # Include both local and global transcript columns so the reader must prefer global pixels.
    pd.DataFrame(
        {
            "fov": [1, 1],
            "cell_ID": [1, 1],
            "x_local_px": [1.0, 2.0],
            "y_local_px": [3.0, 4.0],
            "x_global_px": [2.0, 3.0],
            "y_global_px": [5.0, 6.0],
            "target": ["ACTB", "SystemControl1"],
        }
    ).to_csv(root / "coad_tx_file.csv", index=False)

    # FOV origins are translations, not estimated affines.
    pd.DataFrame({"fov": [1], "x_global_px": [1.0], "y_global_px": [2.0]}).to_csv(
        root / "coad_fov_positions_file.csv",
        index=False,
    )

    pd.DataFrame({"cell_ID": [1, 2], "fov": [1, 1], "ACTB": [2, 1], "SystemControl1": [4, 3]}).to_csv(
        root / "coad_exprMat_file.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "cell_ID": [1, 2],
            "fov": [1, 1],
            "CenterX_global_px": [2.0, 3.0],
            "CenterY_global_px": [5.0, 6.0],
        }
    ).to_csv(root / "coad_metadata_file.csv", index=False)

    return root


def _write_zarr_stores(images_dir: Path, labels_dir: Path) -> None:
    """Write a small channel-grouped image and multiscale labels Zarr fixture."""
    channel_names = ["DNA", "Membrane", "PanCK"]
    datasets = [
        {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0]}]},
        {"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [2.0, 2.0]}]},
    ]
    multiscales = [{"axes": [{"name": "y"}, {"name": "x"}], "datasets": datasets}]

    image_group = zarr.open_group(images_dir, mode="w")
    for channel_index, channel_name in enumerate(channel_names):
        channel_group = image_group.create_group(channel_name)
        channel_group.attrs["multiscales"] = multiscales
        channel_group.create_dataset("0", shape=(8, 8), chunks=(4, 4), dtype="uint16")
        channel_group.create_dataset("1", shape=(4, 4), chunks=(4, 4), dtype="uint16")
        channel_group["0"][:] = channel_index
        channel_group["1"][:] = channel_index

    labels_group = zarr.open_group(labels_dir, mode="w")
    labels_group.attrs["multiscales"] = multiscales
    labels_group.create_dataset("0", shape=(8, 8), chunks=(4, 4), dtype="uint16")
    labels_group.create_dataset("1", shape=(4, 4), chunks=(4, 4), dtype="uint16")
    labels_group["0"][:] = 1
    labels_group["1"][:] = 1


def test_cosmx_reads_global_transcripts_and_filters_panel_genes(tmp_path):
    dataset_path = _write_cosmx_dataset(tmp_path)

    # Write a one-column panel file in the format supplied for the COAD dataset.
    panel_path = tmp_path / "COAD_panel.csv"
    panel_path.write_text("x\nACTB\n", encoding="utf-8")

    sdata = cosmx(
        dataset_path,
        dataset_id="coad",
        to_coordinate_system="sample",
        keep_gene_names=panel_path,
    )

    # Default reads are images plus one transcript layer; vendor labels and tables stay opt-in.
    assert "image_sample" in sdata.images
    assert "transcripts_sample" in sdata.points
    assert sdata.labels == {}
    assert sdata.tables == {}

    # Confirm control probes are absent and global pixel columns were used as-is.
    points = sdata["transcripts_sample"].compute()
    assert points[_GENES_KEY].tolist() == ["ACTB"]
    assert "target" not in points.columns
    assert points[["x", "y"]].to_numpy().tolist() == [[2.0, 5.0]]
    assert isinstance(get_transformation(sdata["transcripts_sample"], to_coordinate_system="sample"), Identity)

    assert isinstance(get_transformation(sdata["image_sample"], to_coordinate_system="sample"), Identity)


def test_cosmx_allows_global_zarr_without_positions(tmp_path):
    dataset_path = _write_cosmx_dataset(tmp_path)
    (dataset_path / "coad_fov_positions_file.csv").unlink()

    # Preserve the valid identity-coordinate case when the dataset contains only one FOV.
    sdata = cosmx(dataset_path, dataset_id="coad", to_coordinate_system="sample")

    assert isinstance(get_transformation(sdata["image_sample"], to_coordinate_system="sample"), Identity)


def test_cosmx_uses_one_global_zarr_for_multiple_fov_positions(tmp_path):
    dataset_path = _write_cosmx_dataset(tmp_path)

    # Keep multiple FOV origins in the metadata without requiring image filenames to expose them.
    pd.DataFrame({"fov": [1, 2], "x_global_px": [1.0, 100.0], "y_global_px": [2.0, 200.0]}).to_csv(
        dataset_path / "coad_fov_positions_file.csv",
        index=False,
    )

    sdata = cosmx(dataset_path, dataset_id="coad", to_coordinate_system="sample")

    assert list(sdata.images) == ["image_sample"]


def test_cosmx_rejects_legacy_raster_input(tmp_path):
    dataset_path = tmp_path / "legacy"
    images_dir = dataset_path / "CellComposite"
    images_dir.mkdir(parents=True)
    (images_dir / "CellComposite.tif").write_bytes(b"legacy raster")
    pd.DataFrame({"target": ["ACTB"], "x_global_px": [1.0], "y_global_px": [2.0]}).to_csv(
        dataset_path / "coad_tx_file.csv",
        index=False,
    )

    # Require modern OME-Zarr input instead of silently loading legacy raster files.
    with pytest.raises(FileNotFoundError, match="Legacy raster inputs are not supported"):
        cosmx(dataset_path, dataset_id="coad", to_coordinate_system="sample")


def test_cosmx_applies_fov_origins_to_local_transcripts(tmp_path):
    dataset_path = _write_cosmx_dataset(tmp_path)

    # Use local transcript coordinates so the positions CSV is the only placement source.
    pd.DataFrame(
        {
            "fov": [1, 2],
            "x_local_px": [1.0, 2.0],
            "y_local_px": [3.0, 4.0],
            "target": ["ACTB", "ACTB"],
        }
    ).to_csv(dataset_path / "coad_tx_file.csv", index=False)

    # Provide one global origin for each local transcript FOV.
    pd.DataFrame({"fov": [1, 2], "x_global_px": [1.0, 100.0], "y_global_px": [2.0, 200.0]}).to_csv(
        dataset_path / "coad_fov_positions_file.csv",
        index=False,
    )

    sdata = cosmx(dataset_path, dataset_id="coad", to_coordinate_system="sample")
    points = sdata["transcripts_sample"].compute()

    assert points[["x", "y"]].to_numpy().tolist() == [[2.0, 5.0], [102.0, 204.0]]


def test_cosmx_includes_local_transcript_fovs_in_origin_validation(tmp_path):
    dataset_path = _write_cosmx_dataset(tmp_path)

    # Remove global coordinates so the transcript FOV column controls placement.
    pd.DataFrame(
        {
            "fov": [1, 2],
            "x_local_px": [1.0, 2.0],
            "y_local_px": [3.0, 4.0],
            "target": ["ACTB", "ACTB"],
        }
    ).to_csv(dataset_path / "coad_tx_file.csv", index=False)

    # Reject the second transcript FOV before local coordinates can be treated as global.
    with pytest.raises(ValueError, match=r"missing origins for FOVs 2"):
        cosmx(dataset_path, dataset_id="coad", to_coordinate_system="sample")


def test_cosmx_rejects_non_finite_transcript_coordinates(tmp_path):
    dataset_path = _write_cosmx_dataset(tmp_path)

    # Keep the global coordinate schema but introduce an invalid spatial value.
    pd.DataFrame(
        {
            "x_global_px": [np.nan],
            "y_global_px": [2.0],
            "target": ["ACTB"],
        }
    ).to_csv(dataset_path / "coad_tx_file.csv", index=False)

    # Reject invalid points before registering the transcript layer.
    with pytest.raises(ValueError, match="contains non-finite coordinates"):
        cosmx(dataset_path, dataset_id="coad", to_coordinate_system="sample")


def test_cosmx_warns_for_unsupported_image_model_kwargs(tmp_path, caplog):
    dataset_path = _write_cosmx_dataset(tmp_path)

    # Warn about ignored model options while continuing with supported reader arguments.
    cosmx(
        dataset_path,
        dataset_id="coad",
        to_coordinate_system="sample",
        image_models_kwargs={"unsupported": True},
    )

    assert "Ignoring unsupported 'image_models_kwargs' keys: unsupported" in caplog.text


def test_cosmx_reads_optional_labels_and_table(tmp_path):
    dataset_path = _write_cosmx_dataset(tmp_path)
    panel_path = tmp_path / "COAD_panel.csv"
    panel_path.write_text("x\nACTB\n", encoding="utf-8")

    sdata = cosmx(
        dataset_path,
        dataset_id="coad",
        to_coordinate_system="sample",
        keep_gene_names=panel_path,
        cells_table=True,
    )

    # Confirm the vendor labels remain a single global pyramid in global coordinates.
    assert isinstance(get_transformation(sdata["labels_sample"], to_coordinate_system="sample"), Identity)

    table = sdata["table_sample"]
    assert table.var_names.tolist() == ["ACTB"]
    assert table.obs[_REGION_KEY].cat.categories.tolist() == ["labels_sample"]
    assert table.obs[_INSTANCE_KEY].tolist() == [1, 2]

    # Confirm default Sparrow allocation consumes the canonical gene column without extra arguments.
    sdata = allocate(
        sdata,
        labels_layer="labels_sample",
        points_layer="transcripts_sample",
        to_coordinate_system="sample",
        output_layer="allocated",
        update_shapes_layers=False,
        overwrite=True,
    )

    assert sdata["allocated"].var_names.tolist() == ["ACTB"]


def test_cosmx_reads_multiscale_zarr_stores_and_keeps_table_link(tmp_path):
    dataset_path = _write_cosmx_dataset(tmp_path)

    sdata = cosmx(
        dataset_path,
        dataset_id="coad",
        to_coordinate_system="sample",
        cells_table=True,
    )

    image = sdata["image_sample"]
    labels = sdata["labels_sample"]

    # Preserve both vendor pyramid levels while exposing each CellComposite child as a named channel.
    assert isinstance(image, DataTree)
    assert list(image) == ["scale0", "scale1"]
    assert image["scale0"].data_vars["image"].dims == ("c", "y", "x")
    assert image["scale0"].coords["c"].values.tolist() == ["DNA", "Membrane", "PanCK"]
    assert image["scale0"].data_vars["image"].shape == (3, 8, 8)
    assert isinstance(image["scale0"].data_vars["image"].data, Array)

    # Confirm level-zero channel stacking reads the expected values without requiring a full-image compute.
    assert image["scale0"].data_vars["image"].data[:, 0, 0].compute().tolist() == [0, 1, 2]

    # Keep the vendor label pyramid integer-typed and linked to the one global labels layer.
    assert isinstance(labels, DataTree)
    assert list(labels) == ["scale0", "scale1"]
    assert labels["scale0"].data_vars["image"].dtype == np.uint32
    assert sdata["table_sample"].obs[_REGION_KEY].cat.categories.tolist() == ["labels_sample"]
    assert isinstance(get_transformation(image, to_coordinate_system="sample"), Identity)
    assert isinstance(get_transformation(labels, to_coordinate_system="sample"), Identity)


def test_cosmx_writes_zarr_stores_to_backed_output(tmp_path):
    dataset_path = _write_cosmx_dataset(tmp_path / "input")
    output_path = tmp_path / "output.zarr"

    sdata = cosmx(
        dataset_path,
        dataset_id="coad",
        to_coordinate_system="sample",
        cells_labels=True,
        output=output_path,
    )

    # Confirm native Zarr pyramids can use Sparrow's backed layer registration path.
    assert sdata.is_backed()
    assert list(sdata["image_sample"]) == ["scale0", "scale1"]
    assert list(sdata["labels_sample"]) == ["scale0", "scale1"]


def test_cosmx_rejects_empty_gene_panel(tmp_path):
    # Write a panel containing only its header and no gene names.
    panel_path = tmp_path / "empty_panel.csv"
    panel_path.write_text("x\n", encoding="utf-8")

    # Reject the empty whitelist before any CosMx data is loaded.
    with pytest.raises(ValueError, match="does not contain any gene names"):
        _load_keep_gene_names(panel_path)


def test_cosmx_rejects_scalar_gene_name_and_accepts_string_panel_path(tmp_path):
    # Require direct gene filters to be expressed as an iterable rather than a scalar string.
    with pytest.raises(ValueError, match="single gene names are not supported"):
        _load_keep_gene_names("ACTB")

    # Preserve support for string paths to panel files.
    panel_path = tmp_path / "panel.csv"
    panel_path.write_text("gene\nACTB\n", encoding="utf-8")
    assert _load_keep_gene_names(str(panel_path)) == {"ACTB"}

    # Reject empty direct iterables because they would silently remove every transcript.
    with pytest.raises(ValueError, match="at least one gene name"):
        _load_keep_gene_names([])


def test_cosmx_requires_named_labels_directory(tmp_path):
    dataset_path = _write_cosmx_dataset(tmp_path)
    labels_dir = dataset_path / "CellLabels"

    # Remove the named labels directory while leaving the image directory in place.
    shutil.rmtree(labels_dir)

    # Do not reinterpret CellComposite or another FOV folder as a labels directory.
    with pytest.raises(FileNotFoundError, match="labels directory"):
        _discover_files(dataset_path, dataset_id="coad", cells_labels=True, cells_table=False)


def test_cosmx_keeps_related_files_on_the_transcript_dataset_id(tmp_path):
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    (dataset_path / "CellComposite").mkdir()

    # Make the transcript prefix differ from the only available table-file prefix.
    pd.DataFrame({"target": ["ACTB"], "x_global_px": [1.0], "y_global_px": [2.0]}).to_csv(
        dataset_path / "coad_tx_file.csv",
        index=False,
    )
    pd.DataFrame({"cell_ID": [1], "fov": [1], "ACTB": [1]}).to_csv(
        dataset_path / "other_exprMat_file.csv",
        index=False,
    )
    pd.DataFrame({"cell_ID": [1], "fov": [1]}).to_csv(
        dataset_path / "other_metadata_file.csv",
        index=False,
    )

    # Refuse to combine table files belonging to a different dataset prefix.
    with pytest.raises(FileNotFoundError, match="counts/metadata files"):
        _discover_files(dataset_path, dataset_id=None, cells_labels=False, cells_table=True)
