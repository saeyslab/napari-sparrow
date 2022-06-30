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

Then set up a conda virtual environment:

```bash
conda env create -f environment.yml
conda activate napari-spongepy
```

Finally do a local install of the napari plugin:

```
pip install -e .
```

You can then run the plugin by first starting napari, and starting the plugin from napari's menu bar: `napari > Plugins > napari-spongepy`.

This development environment was tested on Windows 11.

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
