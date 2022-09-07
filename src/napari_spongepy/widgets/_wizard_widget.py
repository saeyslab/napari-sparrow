"""
Napari widget for managing the other widgets and giving a general overview of the workflow.
"""

from magicgui.widgets import ComboBox, Container

from napari_spongepy.utils import get_pylogger
from napari_spongepy.widgets import (  # load_widget,
    allocate_widget,
    annotate_widget,
    clean_widget,
    segment_widget,
    visualize_widget,
)

log = get_pylogger(__name__)

current_widget = None


def get_choices():
    return [
        # Use lambdas so that the function is only called when the step is changed and can be removed safely.
        # TODO run call at startup so first widget loads immediately instead of empty option.
        # TODO use Enums
        # ("#0 Load", lambda: load_widget),
        ("#1 Clean", lambda: tuple(["Clean", clean_widget])),
        ("#2 Segment", lambda: tuple(["Segment", segment_widget])),
        ("#3 Allocate", lambda: tuple(["Allocate", allocate_widget])),
        ("#4 Annotate", lambda: tuple(["Annotate", annotate_widget])),
        ("#5 Visualize", lambda: tuple(["Visualize", visualize_widget])),
    ]


# @magic_factory(
#     step={"choices": get_choices},
#     auto_call=True,
# )
def wizard_widget() -> None:
    """
    Napari widget for managing the other widgets and giving a general overview of the workflow.
    TODO add next step button
    """
    # log.debug(step)

    step = ComboBox(label="Step:", choices=get_choices(), name="step")
    clean = clean_widget()
    clean.label = "Clean"
    clean.name = "Clean"
    container = Container(name="global", widgets=[step, clean], labels=False)

    # global current_widget
    # current_widget = clean

    def step_changed(event):
        """This is a callback that updates the current step
        when the step menu selection changes
        """
        name = event()[0]

        # Add widget if not yet exists
        if name not in [x.name for x in container._list]:

            widget = event()[1]()
            widget.label = name
            widget.name = name
            container.append(widget)

        # Hide other widgets
        for widget in list(container):
            if widget.name not in ["step", name]:
                widget.visible = False
            else:
                widget.visible = True

    step.changed.connect(step_changed)

    # global current_widget
    # if current_widget:
    #     viewer.window.remove_dock_widget(current_widget)
    # if step:
    #     current_widget = container.append(
    #         step()())
    #     container.show()
    return container


if __name__ == "__main__":
    print(123)
