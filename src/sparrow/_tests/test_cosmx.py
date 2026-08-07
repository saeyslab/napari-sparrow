from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.models import Labels2DModel, PointsModel, TableModel
from spatialdata.transformations import Affine, Identity, Translation, get_transformation

from sparrow.io._cosmx import _load_keep_gene_names, cosmx
from sparrow.table._allocation import allocate
from sparrow.utils._keys import _GENES_KEY, _INSTANCE_KEY, _REGION_KEY


def _mock_cosmx_sdata() -> SpatialData:
    """Create the minimal upstream-shaped CosMx object needed by the reader test."""
    # Build a point dataframe with one measured gene and one control probe.
    points_data = pd.DataFrame(
        {
            "x_local_px": [1.0, 2.0],
            "y_local_px": [3.0, 4.0],
            "target": ["ACTB", "SystemControl1"],
            "cell_ID": [1, 1],
        }
    )

    # Define the FOV-local-to-global transform that the upstream CosMx reader provides.
    fov_to_global = Affine(
        np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]),
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )

    # Parse the dataframe using the same point schema as spatialdata-io CosMx.
    points = PointsModel.parse(
        points_data,
        coordinates={"x": "x_local_px", "y": "y_local_px"},
        feature_key="target",
        instance_key="cell_ID",
        transformations={"global": fov_to_global},
    )

    # Build a counts table with matching region and instance metadata.
    observations = pd.DataFrame(
        {
            _REGION_KEY: pd.Categorical(["1_labels", "1_labels"]),
            _INSTANCE_KEY: [1, 2],
        },
        index=["1_1", "1_2"],
    )
    variables = pd.DataFrame(index=["ACTB", "SystemControl1"])
    table = AnnData(np.array([[2, 4], [1, 3]]), obs=observations, var=variables)

    # Add SpatialData table metadata using the CosMx region and instance columns.
    table = TableModel.parse(
        table,
        region_key=_REGION_KEY,
        region=["1_labels"],
        instance_key=_INSTANCE_KEY,
    )

    # Add a matching label layer so table-region validation reflects a real CosMx object.
    labels = Labels2DModel.parse(
        np.ones((4, 4), dtype=np.uint32),
        dims=("y", "x"),
        transformations={"global": fov_to_global},
    )

    # Return the same categories that the upstream reader exposes.
    return SpatialData(labels={"1_labels": labels}, points={"1_points": points}, tables={"table": table})


def test_cosmx_filters_panel_genes_and_renames_output(monkeypatch, tmp_path):
    # Write a one-column panel file in the format supplied for the COAD dataset.
    panel_path = tmp_path / "COAD_panel.csv"
    panel_path.write_text("x\nACTB\n", encoding="utf-8")

    # Capture the upstream call while returning a small deterministic CosMx object.
    calls: dict[str, object] = {}

    def fake_cosmx(**kwargs):
        calls.update(kwargs)
        return _mock_cosmx_sdata()

    # Replace only the upstream reader used by Sparrow's wrapper.
    monkeypatch.setattr("sparrow.io._cosmx.sdata_cosmx", fake_cosmx)

    # Read the mocked dataset with the panel and a non-global coordinate-system name.
    sdata = cosmx(
        tmp_path,
        dataset_id="coad",
        to_coordinate_system="sample",
        keep_gene_names=panel_path,
        transcripts=True,
    )

    # Confirm the wrapper forwards the upstream reader arguments unchanged.
    assert calls["path"] == tmp_path
    assert calls["dataset_id"] == "coad"
    assert calls["transcripts"] is True

    # Confirm control probes are absent from the lazily filtered transcript points.
    points = sdata["1_points_sample"].compute()
    assert points[_GENES_KEY].tolist() == ["ACTB"]
    assert "target" not in points.columns
    assert points[["x", "y"]].values.tolist() == [[2.0, 5.0]]
    assert isinstance(get_transformation(sdata["1_points_sample"], to_coordinate_system="sample"), Identity)
    assert isinstance(get_transformation(sdata["1_labels_sample"], to_coordinate_system="sample"), Translation)

    # Confirm the counts table uses the same keep list and renamed label region.
    table = sdata["table_sample"]
    assert table.var_names.tolist() == ["ACTB"]
    assert table.obs[_REGION_KEY].cat.categories.tolist() == ["1_labels_sample"]

    # Confirm default Sparrow allocation consumes the canonical gene column without extra arguments.
    sdata = allocate(
        sdata,
        labels_layer="1_labels_sample",
        points_layer="1_points_sample",
        to_coordinate_system="sample",
        output_layer="allocated",
        update_shapes_layers=False,
        overwrite=True,
    )

    # Confirm the allocated table contains the retained gene.
    assert sdata["allocated"].var_names.tolist() == ["ACTB"]


def test_cosmx_rejects_empty_gene_panel(tmp_path):
    # Write a panel containing only its header and no gene names.
    panel_path = tmp_path / "empty_panel.csv"
    panel_path.write_text("x\n", encoding="utf-8")

    # Reject the empty whitelist before any upstream CosMx data is loaded.
    with pytest.raises(ValueError, match="does not contain any gene names"):
        _load_keep_gene_names(panel_path)
