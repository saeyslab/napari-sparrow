"""
Napari widget for managing the other widgets and giving a general overview of the workflow.
"""
from magicgui import magic_factory
from napari.viewer import Viewer

current_widget = None


@magic_factory(
    step={
        "choices": [
            # Use lambdas so that the function is only called when the step is changed and can be removed safely.
            # TODO run call at startup so first widget loads immediately instead of empty option.
            ("Start Here", None),
            # TODO add rest of widgets
        ]
    },
    auto_call=True,
    layout="horizontal",
)
def visualize_widget(
    viewer: Viewer,
    step,
) -> None:
    if not step:
        return
    global current_widget
    if current_widget:
        viewer.window.remove_dock_widget(current_widget)
    current_widget = viewer.window.add_dock_widget(step()())