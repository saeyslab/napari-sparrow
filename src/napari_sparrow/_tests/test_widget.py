"""This file tests the napari widgets and should be used for development purposes."""
import unittest

import tifffile as tiff
from hydra.core.hydra_config import HydraConfig
from PyQt5.QtCore import QEventLoop

from napari_sparrow import utils as utils
from napari_sparrow.widgets import (
    allocate_widget,
    annotate_widget,
    clean_widget,
    load_widget,
    segment_widget,
)


def test_sparrow_widgets(make_napari_viewer, cfg_pipeline, caplog):
    """
    Integration test for sparrow plugin in napari
    """

    HydraConfig().set_config(cfg_pipeline)

    viewer = make_napari_viewer()

    # Start load widget
    _load_widget = load_widget()

    worker = _load_widget(
        viewer,
        path_image=cfg_pipeline.dataset.image,
        output_dir=cfg_pipeline.paths.output_dir,
    )

    _run_event_loop_until_worker_finishes(worker)

    assert "Finished creating sdata" in caplog.text
    assert f"Added {utils.LOAD}" in caplog.text

    # Start clean widget
    _clean_widget = clean_widget()

    worker = _clean_widget(viewer, viewer.layers[utils.LOAD])

    _run_event_loop_until_worker_finishes(worker)

    assert "Tiling correction finished" in caplog.text
    assert "Min max filtering finished" in caplog.text
    assert "Contrast enhancing finished" in caplog.text
    assert f"Added {utils.CLEAN}" in caplog.text
    assert "Cleaning finished" in caplog.text

    # Start segment widget
    _segment_widget = segment_widget()

    worker = _segment_widget(viewer, viewer.layers[utils.CLEAN])

    _run_event_loop_until_worker_finishes(worker)

    assert "Segmentation finished" in caplog.text
    assert f"Added {utils.SEGMENT}" in caplog.text

    # Start allocate widget
    _allocate_widget = allocate_widget()

    worker = _allocate_widget(viewer, transcripts_file=cfg_pipeline.dataset.coords)

    _run_event_loop_until_worker_finishes(worker)

    assert "Allocation finished" in caplog.text
    assert "Preprocessing AnnData finished" in caplog.text
    assert "Clustering finished" in caplog.text
    assert f"Added '{utils.ALLOCATION}' layer" in caplog.text

    if cfg_pipeline.dataset.markers is not None:

        # Start annotate widget
        _annotate_widget = annotate_widget()

        worker = _annotate_widget(viewer, markers_file=cfg_pipeline.dataset.markers)

        _run_event_loop_until_worker_finishes(worker)

        assert "Scoring genes finished" in caplog.text
        assert "Annotation metadata added" in caplog.text



@unittest.skip
def test_load_widget(make_napari_viewer, cfg_pipeline, caplog):
    """Test if the load works."""
    HydraConfig().set_config(cfg_pipeline)

    viewer = make_napari_viewer()

    _load_widget = load_widget()

    worker = _load_widget(
        viewer,
        path_image=cfg_pipeline.dataset.image,
        output_dir=cfg_pipeline.paths.output_dir,
    )

    _run_event_loop_until_worker_finishes(worker)

    assert "Finished creating sdata" in caplog.text
    assert f"Added {utils.LOAD}" in caplog.text


@unittest.skip
def test_clean_widget(make_napari_viewer, cfg_pipeline, caplog):
    """Tests if the clean widget works."""

    HydraConfig().set_config(cfg_pipeline)

    viewer = make_napari_viewer()

    image = tiff.imread(cfg_pipeline.dataset.image)

    viewer.add_image(image, name=utils.LOAD)

    viewer.layers[utils.LOAD].metadata["cfg"] = cfg_pipeline

    _clean_widget = clean_widget()

    worker = _clean_widget(viewer, viewer.layers[utils.LOAD])

    _run_event_loop_until_worker_finishes(worker)

    assert "Tiling correction finished" in caplog.text
    assert "Min max filtering finished" in caplog.text
    assert "Contrast enhancing finished" in caplog.text
    assert f"Added {utils.CLEAN}" in caplog.text
    assert "Cleaning finished" in caplog.text


def _run_event_loop_until_worker_finishes(worker):
    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    loop.exec_()
