# napari-spongepy

<!-- These badges won't work while the GitHub repo is private:
[![License BSD-3](https://img.shields.io/pypi/l/napari-spongepy.svg?color=green)](https://github.com/saeyslab/napari-spongepy/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-spongepy.svg?color=green)](https://pypi.org/project/napari-spongepy)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-spongepy.svg?color=green)](https://python.org)
[![tests](https://github.com/saeyslab/napari-spongepy/workflows/tests/badge.svg)](https://github.com/saeyslab/napari-spongepy/actions)
[![codecov](https://codecov.io/gh/saeyslab/napari-spongepy/branch/main/graph/badge.svg)](https://codecov.io/gh/saeyslab/napari-spongepy)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-spongepy)](https://napari-hub.org/plugins/napari-spongepy)
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

## Setting up a development environment

First clone the GitHub repo and set it as the current directory:

```bash
git clone https://github.com/saeyslab/napari-spongepy.git
cd napari-spongepy
```

Then set up a conda virtual environment and install the plugin. For GPU support other than CUDA, comment out the `cudatoolkit` line in `environment.yml` and follow the [PyTorch](https://pytorch.org/get-started/locally/) instructions.

```bash
# or Conda
conda env create -f environment.yml
# or Mamba
mamba env update -f environment.yml --prune

conda activate napari-spongepy
pip install -e '.[testing]'
```

Use a local data folder at `data/` or point to a different location by overwriting `paths.data_dir` using the CLI (`... paths.data_dir=/srv/scratch/data/spatial/`) or using an untracked local config at `configs/local/default.yaml`:
```yaml
# @package _global_

paths:
  data_dir: /srv/scratch/data/spatial/
```

## Usage

### napari
You can run the plugin by first starting napari, and starting the plugin from napari's menu bar: `napari > Plugins > napari-spongepy`.

You can also use the napari CLI:
```
napari path/to/image --with napari-spongepy Segment
```

### Jupyter notebooks

Check the notebooks in `experiments`, they also can import the Hydra configs.

### Hydra CLI

Run experiments from the CLI using [Hydra](https://hydra.cc).
```
python src/segment.py --help
```

Run a watershed segmentation on a small amount of test data with:
```
python src/segment.py subset=\'0:100,0:100\' +segmentation=watershed
```
In the log you will see the location of the experiment folder, with the input parameters, logs and output files.

Run both a watershed and a cellpose segmentation on a small amount of test data with:
```
python src/segment.py subset=\'0:100,0:100\' +segmentation={watershed,cellpose} --multirun
```


## Development

This development environment was tested on Windows 11 and CentOS 7 with a NVIDIA GPU and MacOS 12.3 with an M1 Pro.

Install a pre-commit hook to run all configured checks in `.pre-commit-config.yaml`:
```
pre-commit install
pre-commit run --all-files
```

Run the tests in the root of the project:
```
pytest
```

Do a type test:
```
mypy --ignore-missing-imports src/
```

Debug in VS Code using a [Remote Attach](https://code.visualstudio.com/docs/python/debugging#_debugging-by-attaching-over-a-network-connection) `launch.json` and debugpy:
```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Remote Attach",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "."
                }
            ],
            "justMyCode": true
        }
    ]
}
```

```
python -m debugpy --listen 5678 src/pipeline.py
```

## References

- https://github.com/ashleve/lightning-hydra-template
## License

Distributed under the terms of the [BSD-3] license,
"napari-spongepy" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/saeyslab/napari-spongepy/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
