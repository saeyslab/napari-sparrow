import os

from napari_sparrow.plot._plot import plot_image, plot_labels


def test_plot_labels(sdata_multi_c, tmp_path):
    plot_labels(
        sdata_multi_c,
        labels_layer="masks_nuclear",
        output=os.path.join(tmp_path, "labels_nucleus"),
    )

    plot_labels(
        sdata_multi_c,
        labels_layer=["masks_nuclear_aligned", "masks_whole"],
        output=os.path.join(tmp_path, "labels_all"),
        crd=[100, 200, 100, 200],
    )


def test_plot_image(sdata_multi_c, tmp_path):
    plot_image(
        sdata_multi_c,
        img_layer="raw_image",
        channel=[0, 1],
        output=os.path.join(tmp_path, "raw_image"),
    )
