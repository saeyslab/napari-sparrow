# Installation

SPArrOW requires Python 3.11.15 and uses [uv](https://docs.astral.sh/uv/) to create virtual environments and install dependencies. Install uv by following [the official installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

## 1. Clone the repository

```bash
git clone https://github.com/saeyslab/napari-sparrow.git
cd napari-sparrow
```

## 2. Create the environment

Install the required Python version and create a project virtual environment:

```bash
uv python install 3.11.15
uv venv --python 3.11.15
```

## 3. Install `SPArrOW`

Install the package and its core dependencies with:

```bash
uv sync
```

`uv sync` installs the project into the local `.venv` environment. Commands can be run in that environment without activating it by prefixing them with `uv run`.

### Optional dependencies

SPArrOW defines optional dependency sets for specific use cases. Add one or more extras to `uv sync` as needed:

| Use case | Command |
| --- | --- |
| Napari plugin | `uv sync --extra plugin` |
| Unit tests | `uv sync --extra testing` |
| JupyterLab and notebook support | `uv sync --extra notebook` |
| Complete tutorial environment | `uv sync --extra dev` |
| Tiling correction | `uv sync --extra tiling` |
| Command-line interface | `uv sync --extra cli` |
| Documentation | `uv sync --extra docs` |
| `rioxarray` support | `uv sync --extra riox` |

The `dev` extra is a convenience bundle that combines the `testing`, `notebook`, and `riox` extras. It is useful for the complete tutorial collection, but it is not required for the base installation, unit tests alone, or JupyterLab alone.

### Running the tutorial notebooks

This section is optional. A standard SPArrOW installation does not require Git LFS or the tutorial submodule. Follow these steps only if you want to execute the repository's tutorial notebooks locally.

The notebooks are stored in the `docs/tutorials` Git submodule, and the notebook files are managed with [Git LFS](https://git-lfs.com/). Install Git LFS, then initialize the submodule and download the notebook files:

```bash
git lfs install
git submodule update --init --recursive
git -C docs/tutorials lfs pull
```

Install the complete tutorial environment:

```bash
uv sync --extra dev
```

The notebooks can be opened in VS Code or another notebook interface. To launch the optional JupyterLab web interface:

```bash
uv run --no-sync jupyter lab
```

Commands use uv run --no-sync after the relevant uv sync, avoiding redundant extra flags.

To prepare an environment for development and documentation, combine the extras:

```bash
uv sync --extra dev --extra docs
```
