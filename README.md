# sparrow

<!-- These badges won't work while the GitHub repo is private:
[![License BSD-3](https://img.shields.io/pypi/l/napari-sparrow.svg?color=green)](https://github.com/saeyslab/napari-sparrow/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-sparrow.svg?color=green)](https://pypi.org/project/napari-sparrow)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-sparrow.svg?color=green)](https://python.org)
[![tests](https://github.com/saeyslab/napari-sparrow/workflows/tests/badge.svg)](https://github.com/saeyslab/napari-sparrow/actions)
[![codecov](https://codecov.io/gh/saeyslab/napari-sparrow/branch/main/graph/badge.svg)](https://codecov.io/gh/saeyslab/napari-sparrow)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-sparrow)](https://napari-hub.org/plugins/napari-sparrow)
-->

Napari plugin for spatial transcriptomics data analysis

---

This [napari] plugin was generated with [Cookiecutter] using [napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/index.html
-->

## Installation

There are two different installation methods:

1. Using the Python package manager pip (TODO)

```
pip install sparrow (TODO sparrow already in use as library name on PyPi)
```

2. Installation from source

First clone this GitHub repo and set it as the current directory:

```bash
git clone https://github.com/saeyslab/napari-sparrow.git
cd napari-sparrow
```

Depending on your hardware, you may need to adapt the Conda `environment.yml` file as follows:

- On Windows comment out the line `basicpy==1.0.0`. We will install `basicpy` manually, see below.

Now create the conda environment

```bash
# Use standard Conda environment creation
conda env create -f environment.yml
# Or use Mamba as alternative
mamba env update -f environment.yml --prune

conda activate napari-sparrow
```

On Windows one must manually install `basicpy` and `jax` as follows:

```
pip install "jax[cpu]===0.4.10" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver
pip install ml_dtypes==0.2.0
pip install basicpy==1.0.0
```

The manual installation of ml_dtypes used by jax is needed because of [this issue](https://developer.apple.com/forums/thread/737890).

Finally, install `sparrow`

```
pip install -e .
```

## Additional dependencies

To use the plugin, run

```bash
pip install "sparrow[plugin]"
```

or when build from source:

```bash
pip install -e ".[plugin]"
```

To run `sparrow` from the `cli`:

```bash
pip install "sparrow[cli]"
```

or when build from source:

```bash
pip install -e ".[cli]"
```

## Input data

Input data is provided from a [RESOLVE experiment on mouse liver](https://cloud.irc.ugent.be/public/index.php/s/HrXG9WKqjqHBEzS). The dataset used in the examples is mouse liver A1-1. Please download the DAPI-stained image and the .txt file.

## Usage

### Jupyter notebooks

Check the notebooks in `experiments`.

### napari

You can run the plugin by first starting napari, and starting the plugin from napari's menu bar: `napari > Plugins > napari-sparrow`.

Use the plugin to tune the parameters of sparrow for the different steps of the pipeline. Tuning can be done on small crops of the image. After every step, a corresponding configuration _.yaml_ file will be saved in the output directory chosen by the user. We refer to the [hpc](docs/hpc.md) documentation for information on how to use these generated configuration files via the CLI.

### (Hydra) CLI

Run experiments from the CLI using [Hydra](https://hydra.cc). Experiments can be run locally, or on a SLURM cluster.

First copy the `configs` folder (in the root of this repository) locally, and set the paths to the input data and log directory via the _.yaml_ provided at `configs/default.example.yaml`. I.e., first rename `configs/default.example.yaml` to `configs/default.yaml` and update following fields:

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

Next, update `configs/dataset/resolve_liver.yaml`, with the correct path to the input data, relative to _$paths.data_dir_ set earlier, e.g. the fields:

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

Please update the _hydra.searchpath_ with the path to the `configs` folder downloaded locally.

All parameters can also be overwritten from the CLI, e.g. for the size of the min max filter:

```bash
sparrow +experiment=resolve_liver hydra.searchpath="[/Path/to/local/configs]" task_name=results_sparrow clean.size_min_max_filter=35
```

The default values for all parameters for each step of the pipeline can be found at `src/sparrow/configs`.

For more info on configuring experiments, we refer to the [hpc](docs/hpc.md) documentation.

## Contributing

Find more information and instructions in the `docs/` folder.

## References

- https://github.com/ashleve/lightning-hydra-template

## License

Distributed under the terms of the [BSD-3] license,
"sparrow" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/saeyslab/napari-sparrow/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
