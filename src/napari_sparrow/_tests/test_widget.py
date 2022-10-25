"""This file tests the napari widgets and should be used for development purposes."""
from skimage.data import cell
from sparrow.widgets import clean_widget, segment_widget

# # make_napari_viewer is a pytest fixture that returns a napari viewer object
# # capsys is a pytest fixture that captures stdout and stderr output streams
# def test_example_q_widget(make_napari_viewer, capsys):
#     # make viewer and add an image layer using our fixture
#     viewer = make_napari_viewer()
#     viewer.add_image(np.random.random((100, 100)))

#     # create our widget, passing in the viewer
#     my_widget = ExampleQWidget(viewer)

#     # call our widget method
#     my_widget._on_click()

#     # read captured output and check that it's as we expected
#     captured = capsys.readouterr()
#     assert captured.out == "napari has 1 layers\n"


def test_clean_widget(make_napari_viewer, caplog):
    """Tests if the clean widget works."""
    viewer = make_napari_viewer()
    viewer.add_image(cell())

    # this time, our widget will be a MagicFactory or FunctionGui instance
    my_widget = clean_widget()

    # if we "call" this object, it'll execute our function
    my_widget(viewer.layers[0])

    assert "About to clean" in caplog.text


def test_segment_widget(make_napari_viewer, caplog):
    """Test if the segmentation widget works."""
    viewer = make_napari_viewer()
    viewer.add_image(cell())

    # this time, our widget will be a MagicFactory or FunctionGui instance
    my_widget = segment_widget()

    # if we "call" this object, it'll execute our function
    my_widget(viewer.layers[0])

    # read captured logging and check that it's as we expected
    assert "About to segment" in caplog.text
