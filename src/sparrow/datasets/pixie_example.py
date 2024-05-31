from spatialdata import SpatialData


def pixie_example() -> SpatialData:
    """Example pixie dataset."""
    import glob
    import os
    from pathlib import Path

    import dask.array as da
    from dask_image import imread
    from datasets import load_dataset

    import sparrow as sp

    # ['segment_image_data', 'cluster_pixels', 'cluster_cells', 'post_clustering', 'fiber_segmentation', 'LDA_preprocessing', 'LDA_training_inference', 'neighborhood_analysis', 'pairwise_spatial_enrichment', 'ome_tiff', 'ez_seg_data']
    dataset = load_dataset("angelolab/ark_example", "cluster_cells", trust_remote_code=True)

    path_segment_data = Path(dataset["cluster_cells"]["deepcell_output"][0]) / "deepcell_output"

    path_image_data = Path(dataset["cluster_cells"]["image_data"][0]) / "image_data"

    assert path_segment_data.exists(), path_segment_data

    assert path_image_data.exists(), path_image_data

    sdata = SpatialData()

    for i in [0, 1]:
        path = os.path.join(path_image_data, f"fov{i}")
        paths_images = glob.glob(os.path.join(path, "*.tiff"))
        results = []
        channels = []
        for path_image in paths_images:
            channel = os.path.splitext(os.path.basename(path_image))[0]
            channels.append(channel)
            arr = imread.imread(path_image)
            results.append(arr)
        da.stack(results, axis=0)

        sdata = sp.im._add_image_layer(
            sdata,
            arr=da.stack(results, axis=0).squeeze(),
            output_layer=f"raw_image_fov{i}",
            c_coords=channels,
            overwrite=True,
        )

        sdata = sp.im._add_label_layer(
            sdata,
            arr=imread.imread(os.path.join(path_segment_data, f"fov{i}_nuclear.tiff")).squeeze(),
            output_layer=f"label_nuclear_fov{i}",
            overwrite=True,
        )

        sdata = sp.im._add_label_layer(
            sdata,
            arr=imread.imread(os.path.join(path_segment_data, f"fov{i}_whole_cell.tiff")).squeeze(),
            output_layer=f"label_whole_fov{i}",
            overwrite=True,
        )

    return sdata


if __name__ == "__main__":
    pixie_example()
