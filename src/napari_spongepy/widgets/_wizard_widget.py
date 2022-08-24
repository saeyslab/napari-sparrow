"""
Napari widget for managing the other widgets and giving a general overview of the workflow.
"""
from magicgui import magic_factory
from napari.viewer import Viewer

from napari_spongepy.utils import get_pylogger
from napari_spongepy.widgets import (
    allocate_widget,
    annotate_widget,
    clean_widget,
    segmentation_widget,
    visualize_widget,
)

log = get_pylogger(__name__)

current_widget = None


def get_choices(prop_dropdown):
    return [
        # Use lambdas so that the function is only called when the step is changed and can be removed safely.
        # TODO run call at startup so first widget loads immediately instead of empty option.
        ("Start Here", None),
        ("Clean", lambda: clean_widget),
        ("Segment", lambda: segmentation_widget),
        ("Allocate", lambda: allocate_widget),
        ("Annotate", lambda: annotate_widget),
        ("Visualize", lambda: visualize_widget),
    ]


@magic_factory(
    step={"choices": get_choices},
    auto_call=True,
)
def wizard_widget(
    viewer: Viewer,
    step,
) -> None:
    log.debug(step)
    global current_widget
    if current_widget:
        viewer.window.remove_dock_widget(current_widget)
    if step:
        current_widget = viewer.window.add_dock_widget(
            step()(), add_vertical_stretch=True
        )
