# Installation

SPArrOW requires Python 3.11 and uses [uv](https://docs.astral.sh/uv/) to create virtual environments and install dependencies. Install uv by following [the official installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

## Install SPArrOW

The repository currently provides the standard installation workflow from source.

### 1. Clone the repository

```bash
git clone https://github.com/saeyslab/napari-sparrow.git
cd napari-sparrow
```

### 2. Install SPArrOW

`uv sync` reads the project's Python requirement, obtains Python 3.11 if it is not already available, creates the default `.venv` environment, and installs SPArrOW with its core dependencies:

```bash
uv sync
```

Activation is optional because `uv run` uses the project environment automatically. If you want to activate it explicitly, use the command for your shell:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

To use SPArrOW as a Napari plugin, install the `plugin` extra instead:

```bash
uv sync --extra plugin
```

Run commands in the environment without activating it by prefixing them with `uv run`, for example:

```bash
uv run python -c "import sparrow; print(sparrow.__version__)"
```

## Optional dependencies

SPArrOW defines optional dependency sets for specific use cases. Add one or more extras to `uv sync` as needed:

| Use case | Command |
| --- | --- |
| Napari plugin | `uv sync --extra plugin` |
| Focused unit tests | `uv sync --extra testing` |
| JupyterLab and notebook support | `uv sync --extra notebook` |
| Tiling correction | `uv sync --extra tiling` |
| Dask dashboard | `uv sync --extra dashboard` |
| `rioxarray`-backed image I/O | `uv sync --extra riox` |
| Command-line interface | `uv sync --extra cli` |
| Documentation | `uv sync --extra docs` |
| Complete tutorial environment | `uv sync --extra tutorials` |
| Full local test and tutorial environment | `uv sync --extra dev` |

Extras can be combined in one command, for example:

```bash
uv sync --extra plugin --extra cli
```

The `riox` and `dashboard` extras are independent feature extras. `rioxarray` provides an optional Merscope image I/O backend, while Bokeh provides the Dask dashboard. Neither is required by the core package or by ordinary tests.

The `tutorials` extra includes the Napari plugin, notebook tooling, tiling correction, the optional `rioxarray` backend, the Dask dashboard, and the pinned AnnData/Pandas/Squidpy stack used by the tutorials. The `dev` extra combines `testing` and `tutorials` for a complete local test and tutorial environment.

## Running the tutorial notebooks

Tutorial setup is optional. A standard SPArrOW installation does not require Git LFS or the tutorial submodule. Follow these steps only if you want to execute the repository's tutorial notebooks locally.

The notebooks are stored in the `docs/tutorials` Git submodule, and the notebook files are managed with [Git LFS](https://git-lfs.com/). Install Git LFS, then initialize the submodule and download the notebook files:

```bash
git lfs install
git submodule update --init --recursive
git -C docs/tutorials lfs pull
```

Install the complete tutorial environment:

```bash
uv sync --extra tutorials
```

The notebooks can be opened in VS Code or another notebook interface. To launch the optional JupyterLab web interface:

```bash
uv run --no-sync jupyter lab
```

The `SPArrOW_how_to_start` notebook starts a local Dask cluster and prints its dashboard URL. The `dashboard` extra provides the Bokeh dependency needed to open that dashboard in a browser.

## Development and documentation

For development, testing, and documentation, see the [contributing guide](contributing.md). To install the complete local development and documentation environment, combine the extras:

```bash
uv sync --extra dev --extra docs
```
