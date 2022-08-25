"""
Napari widget for managing the other widgets and giving a general overview of the workflow.
"""
from magicgui import magic_factory
from napari.viewer import Viewer

from napari_spongepy.utils import get_pylogger

log = get_pylogger(__name__)

current_widget = None


def get_choices(prop_dropdown):
    from napari_spongepy.widgets import (
        allocate_widget,
        annotate_widget,
        clean_widget,
        segment_widget,
        visualize_widget,
    )

    return [
        # Use lambdas so that the function is only called when the step is changed and can be removed safely.
        # TODO run call at startup so first widget loads immediately instead of empty option.
        # TODO use Enums
        ("#0 Load", None),
        ("#1 Clean", lambda: clean_widget),
        ("#2 Segment", lambda: segment_widget),
        ("#3 Allocate", lambda: allocate_widget),
        ("#4 Annotate", lambda: annotate_widget),
        ("#4 Visualize", lambda: visualize_widget),
    ]


@magic_factory(
    step={"choices": get_choices},
    auto_call=True,
)
def wizard_widget(
    viewer: Viewer,
    step,
) -> None:
    """
    Napari widget for managing the other widgets and giving a general overview of the workflow.
    TODO add next step button
    """
    log.debug(step)
    global current_widget
    if current_widget:
        viewer.window.remove_dock_widget(current_widget)
    if step:
        current_widget = viewer.window.add_dock_widget(
            step()(), add_vertical_stretch=True
        )


if __name__ == "__main__":
    print(123)
