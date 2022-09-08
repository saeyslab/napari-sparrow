"""
Napari widget for managing the other widgets and giving a general overview of the workflow.
"""

from magicgui.widgets import ComboBox, Container, TextEdit
from qtpy.QtWidgets import QSizePolicy

from napari_spongepy.utils import get_pylogger
from napari_spongepy.widgets import (
    allocate_widget,
    annotate_widget,
    clean_widget,
    segment_widget,
    visualize_widget,
)

log = get_pylogger(__name__)


class Step:
    def __init__(self, name, label, widget, description):
        self.name = name
        self.label = label
        self.widget = widget
        self.description = description

    def __str__(self):
        return self.name

    def get_widget(self):
        widget = self.widget()
        widget.name = self.name
        return widget

    def get_description(self):
        description = TextEdit(
            value=self.description, name=self.name + "description", enabled=False
        )
        description.native.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        return description


def get_choices():
    return [
        (
            "Step 1: Clean",
            Step(
                "clean",
                "#1 Clean",
                clean_widget,
                "#1 Cleaning step:\nThis step performs lighting correction and mask inpainting for the black lines, afterwards a tophat filter and contrast clip are applied.",
            ),
        ),
        (
            "Step 2: Segment",
            Step(
                "segment",
                "#2 Segment",
                segment_widget,
                "#2 Segmentation step description",
            ),
        ),
        (
            "Step 3: Allocate",
            Step(
                "Allocate",
                "#3 Allocate",
                allocate_widget,
                "#3 Allocation step description",
            ),
        ),
        (
            "Step 4: Annotate",
            Step(
                "Annotate",
                "#4 Annotate",
                annotate_widget,
                "#4 Annotation step description",
            ),
        ),
        (
            "Step 5: Visualize",
            Step(
                "Visualize",
                "#5 Visualize",
                visualize_widget,
                "#5 Visualisation step description",
            ),
        ),
    ]


def wizard_widget() -> None:
    """
    Napari widget for managing the other widgets and giving a general overview of the workflow.
    TODO add next step button
    """

    step = ComboBox(label="Step:", choices=get_choices(), name="step")
    container = Container(
        name="global",
        widgets=[
            step,
            get_choices()[0][1].get_description(),
            get_choices()[0][1].get_widget(),
        ],
        labels=False,
    )

    def step_changed(event):
        """This is a callback that updates the current step
        when the step menu selection changes
        """
        name = str(event)

        # Add widget if not yet exists
        if name not in [x.name for x in container._list]:
            container.append(event.get_description())
            container.append(event.get_widget())

        # Hide other widgets
        for widget in list(container):
            if widget.name not in ["step", name, name + "description"]:
                widget.visible = False
            else:
                widget.visible = True

    step.changed.connect(step_changed)

    return container
