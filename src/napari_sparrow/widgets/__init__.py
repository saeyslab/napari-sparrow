"""Export all the widgets for the napari plugin."""
__version__ = "0.0.1"

from napari_sparrow.widgets._allocate_widget import allocate_widget
from napari_sparrow.widgets._annotate_widget import annotate_widget
from napari_sparrow.widgets._clean_widget import clean_widget
from napari_sparrow.widgets._export_widget import export_widget
from napari_sparrow.widgets._segment_widget import segment_widget
from napari_sparrow.widgets._wizard_widget import wizard_widget

__all__ = [
    clean_widget,
    segment_widget,
    wizard_widget,
    allocate_widget,
    annotate_widget,
    export_widget,
]