# napari-sparrow

<!-- These badges won't work while the GitHub repo is private:
[![License BSD-3](https://img.shields.io/pypi/l/napari-sparrow.svg?color=green)](https://github.com/saeyslab/napari-sparrow/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-sparrow.svg?color=green)](https://pypi.org/project/napari-sparrow)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-sparrow.svg?color=green)](https://python.org)
[![tests](https://github.com/saeyslab/napari-sparrow/workflows/tests/badge.svg)](https://github.com/saeyslab/napari-sparrow/actions)
[![codecov](https://codecov.io/gh/saeyslab/napari-sparrow/branch/main/graph/badge.svg)](https://codecov.io/gh/saeyslab/napari-sparrow)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-sparrow)](https://napari-hub.org/plugins/napari-sparrow)
-->

Napari plugin for spatial transcriptomics data analysis

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/index.html
-->

## Installation

There are three different installation methods:

1. Using the napari plugin installer (TODO)
```
napari > Plugins > Install/Uninstall Plugins... > Filter on 'napari-sparrow'
Click 'Install' button next to the plugin
```

2. Using the Python package manager pip (TODO)
```
pip install napari-sparrow
```

3. Clone this GitHub repo and set it as the current directory:
```bash
git clone https://github.com/saeyslab/napari-sparrow.git
cd napari-sparrow
```

After cloning, setup a conda virtual environment and install the plugin. For GPU support other than CUDA, comment out the `cudatoolkit` line in `environment.yml` and follow the [PyTorch](https://pytorch.org/get-started/locally/) instructions.

```bash
# Use standard Conda environment creation
conda env create -f environment.yml
# Or use Mamba as alternative
mamba env update -f environment.yml --prune

conda activate napari-sparrow
pip install -e .
```

## Input data

By default, a local data folder at `data/` is expected.
Different locations can be given using the CLI or an extra config file.

CLI:
```bash
sparrow paths.data_dir=/srv/scratch/data/spatial/
```

Extra config file `configs/local/default.yaml`:
```yaml
# @package _global_

paths:
  data_dir: /srv/scratch/data/spatial/
```

## Usage

### napari
You can run the plugin by first starting napari, and starting the plugin from napari's menu bar: `napari > Plugins > napari-sparrow`.

You can also use the napari CLI:
```
napari path/to/image --with napari-sparrow Wizard
```

### Jupyter notebooks

Check the notebooks in `experiments`, they also can import the Hydra configs.

### Hydra CLI

Run experiments from the CLI using [Hydra](https://hydra.cc).
```
sparrow --help
```

Run a cellpose segmentation on the dataset configured in the configs with:
```
sparrow
```
In the log you will see the location of the experiment folder, with the input parameters, logs and output files.

Run a cellpose segmentation on a small amount of test data with:
```
sparrow subset=\'0:2144,0:2144\'
```

Perform multirun segmentation, [see hpc](/docshpc.md), on a dataset with:
```
sparrow -cd configs/ms_melanoma dataset='glob(*A1*)' -m
```

## Contributing

Find more information and development instructions in the `docs/` folder.

## References

- https://github.com/ashleve/lightning-hydra-template
## License

Distributed under the terms of the [BSD-3] license,
"napari-sparrow" is free and open source software

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
