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
    _VENDOR_CELL_ID_COLUMN,
    _discover_files,
    _load_keep_gene_names,
    _read_cosmx_zarr_levels,
    _vendor_label_ids,
    cosmx,
)
from sparrow.table._allocation import allocate
from sparrow.utils._keys import _CELL_INDEX, _GENES_KEY, _INSTANCE_KEY, _REGION_KEY, _SPATIAL


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

    # FOV origins are translations, not estimated affines. The vendor records each tile's top edge
    # in a y-up system, so y=8.0 matches the 8x8 fixture raster and makes raster row = 8.0 - y.
    pd.DataFrame({"fov": [1], "x_global_px": [0.0], "y_global_px": [8.0]}).to_csv(
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


def _write_multiscale_group(path: Path, multiscales: list[dict], level_paths: tuple[str, ...]) -> Path:
    """Write a bare Zarr group carrying the given multiscale attrs and level arrays."""
    group = zarr.open_group(path, mode="w")
    if multiscales is not None:
        group.attrs["multiscales"] = multiscales
    for level_path in level_paths:
        group.create_dataset(level_path, shape=(4, 4), chunks=(4, 4), dtype="uint16")
    return path


_SINGLE_LEVEL_PYRAMID = {"axes": [{"name": "y"}, {"name": "x"}], "datasets": [{"path": "0"}]}


@pytest.mark.parametrize(
    "multiscales, level_paths, kind, match",
    [
        # A valid Zarr group that simply carries no OME-Zarr pyramid descriptor.
        (None, ("0",), "image", "does not contain a readable multiscale OME-Zarr dataset"),
        # Two pyramid descriptors leave no single answer for which one to read.
        (
            [_SINGLE_LEVEL_PYRAMID, _SINGLE_LEVEL_PYRAMID],
            ("0",),
            "labels",
            r"contains 2 multiscale nodes; expected exactly one",
        ),
        # Metadata names a second level, but only level zero is written, so it is dangling.
        (
            [{"axes": [{"name": "y"}, {"name": "x"}], "datasets": [{"path": "0"}, {"path": "1"}]}],
            ("0",),
            "image",
            r"lists pyramid level '1'.*not present in the store",
        ),
        # The second entry carries no "path", so it describes no array in the store at all.
        (
            [{"axes": [{"name": "y"}, {"name": "x"}], "datasets": [{"path": "0"}, {"scale": [2.0, 2.0]}]}],
            ("0",),
            "image",
            r"dataset entry at position 1 with no 'path' key",
        ),
    ],
)
def test_cosmx_reports_unreadable_multiscale_metadata(tmp_path, multiscales, level_paths, kind, match):
    """Report each way a vendor Zarr group can fail to describe exactly one readable pyramid."""
    store = _write_multiscale_group(tmp_path / "store", multiscales=multiscales, level_paths=level_paths)

    with pytest.raises(ValueError, match=match):
        _read_cosmx_zarr_levels(store, kind=kind)


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

    # Confirm control probes are absent and global pixel columns were placed in raster space.
    points = sdata["transcripts_sample"].compute()
    assert points[_GENES_KEY].tolist() == ["ACTB"]
    assert "target" not in points.columns
    assert points["fov"].tolist() == [1]
    assert points["cell_ID"].tolist() == [1]
    # x passes through; y is flipped about the mosaic top edge (8.0 - 5.0 = 3.0) so the points
    # overlay the identity-registered image and label rasters instead of being mirrored.
    assert points[["x", "y"]].to_numpy().tolist() == [[2.0, 3.0]]


def test_cosmx_allows_global_zarr_without_positions(tmp_path, caplog):
    dataset_path = _write_cosmx_dataset(tmp_path)
    (dataset_path / "coad_fov_positions_file.csv").unlink()

    # Preserve the valid identity-coordinate case when the dataset contains only one FOV.
    sdata = cosmx(dataset_path, dataset_id="coad", to_coordinate_system="sample")

    # Without the positions file the mosaic top edge is unknown, so the vendor's upward y cannot
    # be converted. That is allowed, but it must be reported rather than silently misplacing data.
    points = sdata["transcripts_sample"].compute()
    assert points[["x", "y"]].to_numpy().tolist() == [[2.0, 5.0], [3.0, 6.0]]
    assert "vertically mirrored" in caplog.text


def test_cosmx_places_global_transcripts_in_raster_rows(tmp_path):
    """Vendor global pixel y increases upward, so it must be flipped onto downward raster rows."""
    dataset_path = _write_cosmx_dataset(tmp_path)

    # Two transcripts at different heights make the direction of the conversion observable.
    pd.DataFrame(
        {
            "fov": [1, 1],
            "x_global_px": [2.0, 3.0],
            "y_global_px": [1.0, 7.0],
            "target": ["ACTB", "ACTB"],
        }
    ).to_csv(dataset_path / "coad_tx_file.csv", index=False)

    sdata = cosmx(dataset_path, dataset_id="coad", to_coordinate_system="sample")
    points = sdata["transcripts_sample"].compute()

    # The mosaic top edge is 8.0, so the transcript high in vendor y becomes a low raster row.
    assert points[["x", "y"]].to_numpy().tolist() == [[2.0, 7.0], [3.0, 1.0]]


def test_cosmx_places_table_centres_in_the_same_space_as_transcripts(tmp_path):
    """The table's spatial coordinates must be converted exactly like the transcript coordinates."""
    dataset_path = _write_cosmx_dataset(tmp_path)

    sdata = cosmx(dataset_path, dataset_id="coad", to_coordinate_system="sample", cells_table=True)

    # Fixture centres are (2, 5) and (3, 6) in vendor global pixels; the mosaic top edge is 8.0.
    assert sdata["table_sample"].obsm[_SPATIAL].tolist() == [[2.0, 3.0], [3.0, 2.0]]


def test_cosmx_vendor_label_ids_match_szudzik_pairing():
    """The mask ID is Szudzik's elegant pairing of (fov, cell_ID), which is injective."""
    # fov >= cell_ID takes the fov*fov + fov + cell_ID branch.
    assert _vendor_label_ids(np.array([1]), np.array([1])).tolist() == [3]
    assert _vendor_label_ids(np.array([400]), np.array([382])).tolist() == [400 * 401 + 382]
    # fov < cell_ID takes the cell_ID*cell_ID + fov branch.
    assert _vendor_label_ids(np.array([1]), np.array([2])).tolist() == [5]
    assert _vendor_label_ids(np.array([200]), np.array([2054])).tolist() == [2054 * 2054 + 200]

    # Distinct (fov, cell_ID) pairs never collide, which is what makes the key usable as an index.
    fovs, cells = np.meshgrid(np.arange(1, 40), np.arange(1, 40))
    keys = _vendor_label_ids(fovs.ravel(), cells.ravel())
    assert np.unique(keys).size == keys.size


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
            "quality": ["good", "review"],
        }
    ).to_csv(dataset_path / "coad_tx_file.csv", index=False)

    # Provide one global origin for each local transcript FOV.
    pd.DataFrame({"fov": [1, 2], "x_global_px": [1.0, 100.0], "y_global_px": [2.0, 200.0]}).to_csv(
        dataset_path / "coad_fov_positions_file.csv",
        index=False,
    )

    sdata = cosmx(dataset_path, dataset_id="coad", to_coordinate_system="sample")
    points = sdata["transcripts_sample"].compute()

    # Local pixels already run in raster orientation, so each FOV is a plain translation:
    # column = origin_x + local_x, row = (mosaic top edge 200.0 - origin_y) + local_y.
    assert points[["x", "y"]].to_numpy().tolist() == [[2.0, 201.0], [102.0, 4.0]]
    assert points["fov"].tolist() == [1, 2]
    assert points["quality"].tolist() == ["good", "review"]


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

    # Reject the missing transcript FOV during in-flight partition processing.
    with pytest.raises(ValueError, match=r"missing origins for FOVs 2"):
        cosmx(dataset_path, dataset_id="coad", to_coordinate_system="sample")


@pytest.mark.parametrize(
    "transcripts, match",
    [
        # Local coordinates with no FOV column: nothing says which tile to translate them by.
        ({"x_local_px": [1.0], "y_local_px": [3.0], "target": ["ACTB"]}, r"missing a fov column"),
        # Only one of the two global axes; the reader must name the axis it is missing rather than
        # falling through to the local branch and complaining about a local column.
        (
            {"fov": [1], "x_global_px": [1.0], "x_local_px": [1.0], "y_local_px": [3.0], "target": ["ACTB"]},
            r"missing a global y column",
        ),
    ],
)
def test_cosmx_rejects_incomplete_transcript_coordinate_columns(tmp_path, transcripts, match):
    """Name the specific coordinate column that a transcript schema is missing."""
    dataset_path = _write_cosmx_dataset(tmp_path)
    pd.DataFrame(transcripts).to_csv(dataset_path / "coad_tx_file.csv", index=False)

    with pytest.raises(ValueError, match=match):
        cosmx(dataset_path, dataset_id="coad", to_coordinate_system="sample")


def test_cosmx_rejects_local_transcripts_without_fov_origins(tmp_path):
    dataset_path = _write_cosmx_dataset(tmp_path)

    # Remove global coordinates so local transcript positions require FOV origins.
    pd.DataFrame(
        {
            "fov": [1],
            "x_local_px": [1.0],
            "y_local_px": [3.0],
            "target": ["ACTB"],
        }
    ).to_csv(dataset_path / "coad_tx_file.csv", index=False)
    (dataset_path / "coad_fov_positions_file.csv").unlink()

    with pytest.raises(ValueError, match=r"local FOV coordinates.*no FOV positions file"):
        cosmx(dataset_path, dataset_id="coad", to_coordinate_system="sample")


def test_cosmx_rejects_non_finite_transcript_coordinates(tmp_path):
    """A transcript with no finite position would be placed arbitrarily, so it must be rejected."""
    dataset_path = _write_cosmx_dataset(tmp_path)

    # Keep the global coordinate schema but introduce an invalid spatial value.
    pd.DataFrame(
        {
            "x_global_px": [np.nan, 3.0],
            "y_global_px": [2.0, 4.0],
            "target": ["ACTB", "ACTB"],
        }
    ).to_csv(dataset_path / "coad_tx_file.csv", index=False)

    # The check lives inside the Dask graph, so it fires as the partition is read rather than
    # forcing the whole transcript frame to be materialized up front.
    with pytest.raises(ValueError, match=r"non-finite x or y coordinate"):
        cosmx(dataset_path, dataset_id="coad", to_coordinate_system="sample")


@pytest.mark.parametrize(
    "file_name, kind, cells_table, tail",
    [
        # Cut off two fields into a new record, the way an interrupted download or copy leaves it.
        ("coad_tx_file.csv", "transcript", False, "1,1"),
        ("coad_fov_positions_file.csv", "FOV positions", False, "1"),
        # Counts and metadata are only parsed, and therefore only checked, when the table is requested.
        ("coad_exprMat_file.csv", "counts", True, "1,1"),
        ("coad_metadata_file.csv", "metadata", True, "1,1"),
    ],
)
def test_cosmx_rejects_truncated_csv_before_writing_output(tmp_path, file_name, kind, cells_table, tail):
    """A vendor CSV cut off mid-line would be padded with NaN by pandas, so it must be rejected up front."""
    dataset_path = _write_cosmx_dataset(tmp_path / "input")
    output_path = tmp_path / "output.zarr"

    # Append the tail without the line break that a completed write ends with.
    with (dataset_path / file_name).open("a", encoding="utf-8", newline="") as handle:
        handle.write(tail)

    with pytest.raises(ValueError, match=rf"CosMx {kind} file .* appears to be truncated"):
        cosmx(
            dataset_path,
            dataset_id="coad",
            to_coordinate_system="sample",
            cells_table=cells_table,
            output=output_path,
        )

    # The check runs while inputs are validated, so no partial output store is left behind.
    assert not output_path.exists()


def test_cosmx_rejects_zero_padded_transcript_csv(tmp_path):
    """A preallocated download that stopped early ends in zero padding rather than a partial line."""
    dataset_path = _write_cosmx_dataset(tmp_path)

    # Pad well past both the first tail window and the csv field size limit, so the last "line" is
    # one oversized field that must still be recognized as incomplete.
    with (dataset_path / "coad_tx_file.csv").open("ab") as handle:
        handle.write(b"\x00" * 200_000)

    with pytest.raises(ValueError, match=r"appears to be truncated: its last line has 0 of the header's 7 fields"):
        cosmx(dataset_path, dataset_id="coad", to_coordinate_system="sample")


def test_cosmx_warns_for_csv_without_final_line_break(tmp_path, caplog):
    """A complete last line without a final line break may be legitimate, so it only warns."""
    dataset_path = _write_cosmx_dataset(tmp_path)
    transcripts_path = dataset_path / "coad_tx_file.csv"

    # Drop the final line break but keep every field of the last line.
    transcripts_path.write_bytes(transcripts_path.read_bytes().rstrip(b"\r\n"))

    sdata = cosmx(dataset_path, dataset_id="coad", to_coordinate_system="sample")

    assert "does not end with a line break" in caplog.text
    # The last transcript is still read, including its final field.
    assert sdata["transcripts_sample"].compute()[_GENES_KEY].tolist() == ["ACTB", "SystemControl1"]


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

    table = sdata["table_sample"]
    assert table.var_names.tolist() == ["ACTB"]
    assert table.obs[_REGION_KEY].cat.categories.tolist() == ["labels_sample"]
    # The instance key is the integer label value the vendor stores in CellLabels, paired from
    # (fov, cell_ID): pair(1, 1) = 1*1 + 1 + 1 = 3 and pair(1, 2) = 2*2 + 1 = 5.
    assert table.obs[_INSTANCE_KEY].tolist() == [3, 5]
    # The vendor's per-FOV cell number stays available under its own column.
    assert table.obs[_VENDOR_CELL_ID_COLUMN].tolist() == [1, 2]

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


def test_cosmx_disambiguates_cell_ids_repeated_across_fovs(tmp_path):
    """The vendor per-FOV cell_ID restarts at 1 in every FOV, so two different cells that happen
    to share the same local cell_ID in different FOVs must not be merged into one observation."""
    dataset_path = _write_cosmx_dataset(tmp_path)

    # Two distinct cells colliding on the vendor's per-FOV cell_ID (both "1") but belonging to
    # different FOVs, with different gene counts so an accidental merge would be detectable.
    pd.DataFrame({"cell_ID": [1, 1], "fov": [1, 2], "ACTB": [5, 9], "SystemControl1": [0, 0]}).to_csv(
        dataset_path / "coad_exprMat_file.csv", index=False
    )
    pd.DataFrame(
        {
            "cell_ID": [1, 1],
            "fov": [1, 2],
            "CenterX_global_px": [2.0, 3.0],
            "CenterY_global_px": [5.0, 6.0],
        }
    ).to_csv(dataset_path / "coad_metadata_file.csv", index=False)

    sdata = cosmx(dataset_path, dataset_id="coad", to_coordinate_system="sample", cells_table=True)

    table = sdata["table_sample"]
    # Both cells are kept as distinct observations. The pairing is injective, so two cells that
    # share cell_ID=1 in different FOVs get different keys: pair(1, 1) = 3 and pair(2, 1) = 7.
    assert table.n_obs == 2
    assert table.obs[_INSTANCE_KEY].tolist() == [3, 7]
    assert table.obs.index.name == _CELL_INDEX
    # Counts are not merged between the two same-local-ID cells.
    assert table[:, "ACTB"].X.toarray().ravel().tolist() == [5, 9]


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

    # Every layer is registered with an identity transform, because the reader expresses all
    # coordinates in the mosaic's own raster pixel space rather than transforming into it.
    for layer in ("image_sample", "labels_sample", "transcripts_sample"):
        assert isinstance(get_transformation(sdata[layer], to_coordinate_system="sample"), Identity)


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


def test_cosmx_validates_gene_whitelists(tmp_path):
    """Accept panel paths and iterables, and reject every input that would filter out everything."""
    # Require direct gene filters to be expressed as an iterable rather than a scalar string.
    with pytest.raises(ValueError, match="single gene names are not supported"):
        _load_keep_gene_names("ACTB")

    # Preserve support for string paths to panel files.
    panel_path = tmp_path / "panel.csv"
    panel_path.write_text("gene\nACTB\n", encoding="utf-8")
    assert _load_keep_gene_names(str(panel_path)) == {"ACTB"}

    # Reject a header-only panel, which would silently remove every transcript.
    empty_panel_path = tmp_path / "empty_panel.csv"
    empty_panel_path.write_text("x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not contain any gene names"):
        _load_keep_gene_names(empty_panel_path)

    # Reject empty direct iterables for the same reason.
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


def test_cosmx_rejects_duplicate_vendor_cell_keys(tmp_path):
    dataset_path = _write_cosmx_dataset(tmp_path)

    # Repeat one fov+cell_ID pair, which would otherwise keep only the last colliding row and
    # leave a non-unique observation index behind.
    pd.DataFrame(
        {
            "cell_ID": [1, 1],
            "fov": [1, 1],
            "CenterX_global_px": [2.0, 3.0],
            "CenterY_global_px": [5.0, 6.0],
        }
    ).to_csv(dataset_path / "coad_metadata_file.csv", index=False)

    with pytest.raises(ValueError, match=r"duplicate fov\+cell ID key\(s\), e.g. 1_1"):
        cosmx(dataset_path, dataset_id="coad", to_coordinate_system="sample", cells_table=True)


def test_cosmx_table_keeps_only_cells_present_in_both_counts_and_metadata(tmp_path):
    dataset_path = _write_cosmx_dataset(tmp_path)

    # cell 1 is in both files, cell 2 is metadata-only and cell 99 is counts-only.
    pd.DataFrame({"cell_ID": [1, 99], "fov": [1, 1], "ACTB": [7, 3], "SystemControl1": [0, 0]}).to_csv(
        dataset_path / "coad_exprMat_file.csv", index=False
    )
    pd.DataFrame(
        {
            "cell_ID": [1, 2],
            "fov": [1, 1],
            "CenterX_global_px": [2.0, 3.0],
            "CenterY_global_px": [5.0, 6.0],
        }
    ).to_csv(dataset_path / "coad_metadata_file.csv", index=False)

    sdata = cosmx(dataset_path, dataset_id="coad", to_coordinate_system="sample", cells_table=True)

    table = sdata["table_sample"]
    # Only the intersection survives, and its counts land on the right row.
    assert table.obs[_INSTANCE_KEY].tolist() == [3]
    assert table[:, "ACTB"].X.toarray().ravel().tolist() == [7]


def test_cosmx_rejects_malformed_counts_identifiers(tmp_path):
    """A blank identifier must fail loudly instead of silently dropping the chunk that holds it.

    ``read_csv`` infers dtypes independently per chunk, so one blank value turns that chunk's
    identifier column into float64. Its keys would then match no metadata cell and every cell in
    the chunk would vanish from the table without any error.
    """
    dataset_path = _write_cosmx_dataset(tmp_path)

    # Leave the fov of the second counts row empty.
    (dataset_path / "coad_exprMat_file.csv").write_text(
        "cell_ID,fov,ACTB,SystemControl1\n1,1,2,4\n2,,1,3\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Integer column has NA values"):
        cosmx(dataset_path, dataset_id="coad", to_coordinate_system="sample", cells_table=True)


def test_cosmx_does_not_read_a_redundant_id_alias_as_a_gene(tmp_path):
    """A counts file carrying two identifier aliases must not turn the unused one into a gene."""
    dataset_path = _write_cosmx_dataset(tmp_path)

    # Carry both 'cell_ID' and its 'cell_id' alias, as some vendor exports do.
    pd.DataFrame(
        {
            "cell_ID": [1, 2],
            "cell_id": [1, 2],
            "fov": [1, 1],
            "ACTB": [2, 1],
            "SystemControl1": [4, 3],
        }
    ).to_csv(dataset_path / "coad_exprMat_file.csv", index=False)

    sdata = cosmx(dataset_path, dataset_id="coad", to_coordinate_system="sample", cells_table=True)

    # The unused alias is an identifier, not a measured gene, so it must not reach var.
    assert sdata["table_sample"].var_names.tolist() == ["ACTB", "SystemControl1"]


def test_cosmx_table_keeps_cells_whose_counts_are_all_zero(tmp_path):
    dataset_path = _write_cosmx_dataset(tmp_path)

    # A cell with no detected transcripts is still a cell, so it must stay in the table.
    pd.DataFrame({"cell_ID": [1, 2], "fov": [1, 1], "ACTB": [0, 4], "SystemControl1": [0, 0]}).to_csv(
        dataset_path / "coad_exprMat_file.csv", index=False
    )

    sdata = cosmx(dataset_path, dataset_id="coad", to_coordinate_system="sample", cells_table=True)

    table = sdata["table_sample"]
    assert table.obs[_INSTANCE_KEY].tolist() == [3, 5]
    assert table[:, "ACTB"].X.toarray().ravel().tolist() == [0, 4]
