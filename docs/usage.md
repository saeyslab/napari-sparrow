# Usage

## Input data

Input data is provided from a [RESOLVE experiment on mouse liver](https://cloud.irc.ugent.be/public/index.php/s/HrXG9WKqjqHBEzS). The dataset used in the examples is mouse liver A1-1. Please download the DAPI-stained image and the .txt file.

### Jupyter notebooks

Check the notebooks in `experiments`.

### napari
You can run the plugin by first starting napari, and starting the plugin from napari's menu bar: `napari > Plugins > napari-sparrow`.

Use the plugin to tune the parameters of sparrow for the different steps of the pipeline. Tuning can be done on small crops of the image. After every step, a corresponding configuration *.yaml* file will be saved in the output directory chosen by the user. We refer to the [hpc](hpc.md) documentation for information on how to use these generated configuration files via the CLI.

### (Hydra) CLI

Run experiments from the CLI using [Hydra](https://hydra.cc). Experiments can be run locally, or on a SLURM cluster.

First copy the `configs` folder (in the root of this repository) locally, and set the paths to the input data and log directory via the *.yaml* provided at `configs/default.example.yaml`. I.e., first rename `configs/default.example.yaml` to `configs/default.yaml` and update following fields:

```yaml
paths:
  log_dir: ${oc.env:HOME}/VIB/DATA/logs/
  data_dir: ${oc.env:HOME}/VIB/DATA/
```

When running locally, use the following setting for the hydra launcher:

```yaml
defaults:
  - override /hydra/launcher: submitit_local
```

If sparrow is run on a SLURM cluster, change this to:

```yaml
defaults:
  - override /hydra/launcher: submitit_slurm
```

Next, update `configs/dataset/resolve_liver.yaml`, with the correct path to the input data, relative to *$paths.data_dir* set earlier, e.g. the fields:

```yaml
data_dir: ${paths.data_dir}/resolve/resolve_liver
image: ${dataset.data_dir}/20272_slide1_A1-1_DAPI.tiff
coords: ${dataset.data_dir}/20272_slide1_A1-1_results.txt
markers: ${dataset.data_dir}/markerGeneListMartinNoLow.csv
```
assuming the RESOLVE mouse liver data is used.

The RESOLVE mouse liver experiment is preconfigured in `configs/experiment/resolve_liver.yaml`, and can now be run from the CLI:

```bash
sparrow +experiment=resolve_liver hydra.searchpath="[/Path/to/local/configs]" task_name=results_sparrow
```

Please update the *hydra.searchpath* with the path to the `configs` folder downloaded locally.

All parameters can also be overwritten from the CLI, e.g. for the size of the min max filter:

```bash
sparrow +experiment=resolve_liver hydra.searchpath="[/Path/to/local/configs]" task_name=results_sparrow clean.size_min_max_filter=35
```

The default values for all parameters for each step of the pipeline can be found at `src/napari_sparrow/configs`.