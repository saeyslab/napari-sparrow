from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from spatialdata.transformations import Identity, Translation, get_transformation

from sparrow.io._cosmx import (
    _discover_files,
    _fov_translation,
    _load_keep_gene_names,
    cosmx,
)
from sparrow.table._allocation import allocate
from sparrow.utils._keys import _GENES_KEY, _INSTANCE_KEY, _REGION_KEY


def _write_cosmx_dataset(root: Path) -> Path:
    """Write a minimal CosMx export that the native reader can load."""
    images_dir = root / "CellComposite"
    labels_dir = root / "CellLabels"
    images_dir.mkdir()
    labels_dir.mkdir()

    # Write one grayscale FOV image and a matching integer label mask.
    tifffile.imwrite(images_dir / "CellComposite_F001.tif", np.arange(64, dtype=np.uint8).reshape(8, 8))
    tifffile.imwrite(labels_dir / "CellLabels_F001.tif", np.ones((8, 8), dtype=np.uint16))

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
    assert "1_image_sample" in sdata.images
    assert "transcripts_sample" in sdata.points
    assert sdata.labels == {}
    assert sdata.tables == {}

    # Confirm control probes are absent and global pixel columns were used as-is.
    points = sdata["transcripts_sample"].compute()
    assert points[_GENES_KEY].tolist() == ["ACTB"]
    assert "target" not in points.columns
    assert points[["x", "y"]].to_numpy().tolist() == [[2.0, 5.0]]
    assert isinstance(get_transformation(sdata["transcripts_sample"], to_coordinate_system="sample"), Identity)

    image_transform = get_transformation(sdata["1_image_sample"], to_coordinate_system="sample")
    assert isinstance(image_transform, Translation)
    assert np.allclose(image_transform.translation, [1.0, 2.0])


def test_cosmx_allows_single_fov_without_positions(tmp_path):
    dataset_path = _write_cosmx_dataset(tmp_path)
    (dataset_path / "coad_fov_positions_file.csv").unlink()

    # Preserve the valid identity-coordinate case when the dataset contains only one FOV.
    sdata = cosmx(dataset_path, dataset_id="coad", to_coordinate_system="sample")

    assert isinstance(get_transformation(sdata["1_image_sample"], to_coordinate_system="sample"), Identity)


def test_cosmx_rejects_missing_positions_for_multiple_fovs(tmp_path):
    dataset_path = _write_cosmx_dataset(tmp_path)

    # Add a second FOV so identity transforms would place two images on top of each other.
    tifffile.imwrite(
        dataset_path / "CellComposite" / "CellComposite_F002.tif",
        np.zeros((8, 8), dtype=np.uint8),
    )
    (dataset_path / "coad_fov_positions_file.csv").unlink()

    # Reject ambiguous geometry before image or transcript layers are registered.
    with pytest.raises(ValueError, match=r"multiple FOVs.*no FOV positions file"):
        cosmx(dataset_path, dataset_id="coad", to_coordinate_system="sample")


def test_cosmx_rejects_incomplete_positions_for_multiple_fovs(tmp_path):
    dataset_path = _write_cosmx_dataset(tmp_path)

    # Add a second FOV while leaving the positions file with only the first origin.
    tifffile.imwrite(
        dataset_path / "CellComposite" / "CellComposite_F002.tif",
        np.zeros((8, 8), dtype=np.uint8),
    )

    # Reject a partial origin mapping instead of assigning the missing FOV identity coordinates.
    with pytest.raises(ValueError, match=r"missing origins for FOVs 2"):
        cosmx(dataset_path, dataset_id="coad", to_coordinate_system="sample")


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

    # Confirm vendor labels received the same FOV translation as the image.
    assert isinstance(get_transformation(sdata["1_labels_sample"], to_coordinate_system="sample"), Translation)

    table = sdata["table_sample"]
    assert table.var_names.tolist() == ["ACTB"]
    assert table.obs[_REGION_KEY].cat.categories.tolist() == ["1_labels_sample"]
    assert table.obs[_INSTANCE_KEY].tolist() == [1, 2]

    # Confirm default Sparrow allocation consumes the canonical gene column without extra arguments.
    sdata = allocate(
        sdata,
        labels_layer="1_labels_sample",
        points_layer="transcripts_sample",
        to_coordinate_system="sample",
        output_layer="allocated",
        update_shapes_layers=False,
        overwrite=True,
    )

    assert sdata["allocated"].var_names.tolist() == ["ACTB"]


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


def test_cosmx_rejects_unsupported_fov_transformation():
    # Report unsupported transform values before they reach SpatialData layer registration.
    with pytest.raises(ValueError, match="unsupported transformation type 'object'"):
        _fov_translation("1", {"1": object()})


def test_cosmx_requires_named_labels_directory(tmp_path):
    dataset_path = _write_cosmx_dataset(tmp_path)
    labels_dir = dataset_path / "CellLabels"

    # Remove the named labels directory while leaving the image directory in place.
    for label_path in labels_dir.iterdir():
        label_path.unlink()
    labels_dir.rmdir()

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
