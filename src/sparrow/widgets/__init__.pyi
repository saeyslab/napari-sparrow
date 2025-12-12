"""Export all the widgets for the napari plugin."""

__version__ = "0.0.1"

from ._allocate_widget import allocate_widget
from ._annotate_widget import annotate_widget
from ._clean_widget import clean_widget
from ._load_widget import load_widget
from ._segment_widget import segment_widget
from ._wizard_widget import wizard_widget

__all__ = [
    "load_widget",
    "clean_widget",
    "segment_widget",
    "wizard_widget",
    "allocate_widget",
    "annotate_widget",
]
