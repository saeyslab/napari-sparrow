![alt text](https://github.com/saeyslab/napari-sparrow/blob/main/docs/_static/img/logo.png)

<!-- These badges won't work while the GitHub repo is private:
[![License BSD-3](https://img.shields.io/pypi/l/harpy.svg?color=green)](https://github.com/saeyslab/harpy/raw/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/harpy-analysis.svg?color=green)](https://python.org)
[![codecov](https://codecov.io/gh/saeyslab/harpy/graph/badge.svg?token=7UXMDWVYFZ)](https://codecov.io/gh/saeyslab/harpy)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/harpy)](https://napari-hub.org/plugins/harpy)
-->

# **SPArrOW: A library for Spatial Transcriptomics Data Analysis.**

[![documentation badge](https://readthedocs.org/projects/sparrow-pipeline/badge/?version=latest)](https://sparrow-pipeline.readthedocs.io/en/latest/)

Note: This package is still under active development.

## Installation

**Recommended** for end-users. Install the latest `harpy-analysis` [PyPI package](https://pypi.org/project/harpy-analysis) with the `extra` dependencies in a local Python environment.

```bash
uv venv --python=3.12 # set python version
source .venv/bin/activate # activate the virtual environment
uv pip install 'harpy-analysis[extra]' # use uv to pip install dependencies
python -c 'import harpy; print(harpy.__version__)' # check if the package is installed
```

**Only for developers.** Clone this repository locally, install the `.[dev]` instead of the `[extra]` dependencies and read the contribution guide.

```bash
# Clone repository from GitHub
uv venv --python=3.12 # set python version
source .venv/bin/activate # activate the virtual environment
uv pip install -e '.[dev]' # use uv to pip install dependencies
python -c 'import harpy; print(harpy.__version__)' # check if the package is installed
# make changes
python -m pytest # run the tests
```

Checkout the docs for [installation instructions](https://github.com/saeyslab/harpy/blob/main/docs/installation.md) using [conda](https://github.com/conda/conda).

## Tutorials

Tutorials are available [here](https://sparrow-pipeline.readthedocs.io/en/latest/).

## Usage

[Learn](docs/usage.md) how `SPArrOW` can be integrated into your workflow in different ways.

## Contributing

See [here](docs/contributing.md) for info on how to contribute to `SPArrOW`.

## References

- https://github.com/ashleve/lightning-hydra-template

## License

Check license under license. SPArrOW is free for academic usage.
For commercial usage, please contact Saeyslab.

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/saeyslab/harpy/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

## Cite us

Our article currently is available as preprint. You can cite us as follows: 
@article{pollaris2024sparrow,

  title={SPArrOW: a flexible, interactive and scalable pipeline for spatial transcriptomics analysis},
  
  author={Pollaris, Lotte and Vanneste, Bavo and Rombaut, Benjamin and Defauw, Arne and Vernaillen, Frank and Mortier, Julien and Vanhenden, Wout and Martens, Liesbet and Thon{\'e}, Tinne and Hastir, Jean-Francois and others},
  
  journal={bioRxiv},
  
  pages={2024--07},
  
  year={2024},
  
  publisher={Cold Spring Harbor Laboratory}
}

Code to replicate the analysis of the paper can be found here: https://github.com/lopollar/SPArrOW_scripts
