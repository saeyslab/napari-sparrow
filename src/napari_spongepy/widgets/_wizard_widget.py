"""
Napari widget for managing the other widgets and giving a general overview of the workflow.
"""


from magicgui.widgets import ComboBox, Container, Label, TextEdit
from qtpy.QtGui import QPixmap
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


# Class for step widgets
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
            value=self.description,
            name=self.name + "description",
            enabled=True,
        )
        description.native.setReadOnly(True)
        # description.min_height = 250
        description.native.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        description.native.setMarkdown(self.description)
        return description


# Step as choices
def get_choices():
    return [
        (
            "Step 1: Clean",
            Step(
                "clean",
                "#1 Clean",
                clean_widget,
                """## Step 1: Cleaning\n### Improves image quality:\n- Select an image\n- Select a subset of the image by adding a rectangular shapes layer\n - The tilesize of the image\n - The contrast adjusment based on OpenCV CLAHE\n - The tophat filter improves object separation""",
            ),
        ),
        (
            "Step 2: Segment",
            Step(
                "segment",
                "#2 Segment",
                segment_widget,
                """## Step 2: Segmentation\n### Segments the cells from the image:\n- Select an image\n - Select a subset of the image by adding a rectangular shapes layer\n - Select a device, either cpu or cuda:0\n - Minimum amount of pixels in mask\n - Shape of the mask filter, higher means less round\n - Mean expected diameter\n - Threshold amount mask kept, smaller means more\n - Model type\n - Channel selection for images with multiple channels""",
            ),
        ),
        (
            "Step 3: Allocate",
            Step(
                "Allocate",
                "#3 Allocate",
                allocate_widget,
                """## Step 3: Allocation\n### Consists of four subprocesses:\n - CreateAdata: Extracts the shapes of the cells and reads in the transcript file (.txt).\n - PreprocessAdata: This step calculates the QC metrics and performs normalization based on the size.\n - FilterOnSize: This step filters out any cells that fall outside the min-max size range. \n - Clustering: This step performs neighborhood analysis and leiden clustering.""",
            ),
        ),
        (
            "Step 4: Annotate",
            Step(
                "Annotate",
                "#4 Annotate",
                annotate_widget,
                """## Step 4: Annotation\n### Consists of one subprocess:\n - ScoreGenes: This step annotates the cells based on the marker geneslist (.csv).""",
            ),
        ),
        (
            "Step 5: Visualize",
            Step(
                "Visualize",
                "#5 Visualize",
                visualize_widget,
                """## Step 5 Visualisation:\n### Consists of three subprocesses:\n - ClusterCleanliness: This step checks how well the clusters agree with the celltyping.\n - Enrichment: This step shows the enrichment between the different celltypes.\n - SaveData: This step saves the shapes objects as geojson and the AnnData in the h5ad file of the given folder.""",
            ),
        ),
    ]


def wizard_widget() -> None:
    """
    Napari widget for managing the other widgets and giving a general overview of the workflow.
    """

    # Set DaMBi Icon
    icon = Label(name="icon", value="Made by DaMBi")
    pixmap = QPixmap("./src/napari_spongepy/widgets/dambi-white.png")
    icon.native.setPixmap(pixmap)

    # Step selector
    step = ComboBox(label="Step:", choices=get_choices(), name="step")

    # Global container holds all widgets
    container = Container(
        name="global",
        widgets=[
            icon,
            step,
            get_choices()[0][1].get_description(),
            get_choices()[0][1].get_widget(),
        ],
        labels=False,
        layout="vertical",
    )
    container.native.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

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
            if widget.name not in ["icon", "step", name, name + "description"]:
                widget.visible = False
            else:
                widget.visible = True

    step.changed.connect(step_changed)

    return container
