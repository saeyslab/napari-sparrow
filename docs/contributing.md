# Development

## Setting up a development environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and clone the GitHub repository:

```bash
git clone https://github.com/saeyslab/napari-sparrow.git
cd napari-sparrow
```

The project requires Python 3.11.15. Create a dedicated development environment and install the full local validation bundle:

```bash
uv python install 3.11.15
uv venv --python 3.11.15 .venv-dev
export UV_PROJECT_ENVIRONMENT=.venv-dev
uv sync --extra dev
```

The `dev` extra combines the focused test tools with the complete `tutorials` environment. That includes the Napari plugin, notebook support, tiling correction, the optional `rioxarray` image backend, and the Bokeh-powered Dask dashboard. Keep `UV_PROJECT_ENVIRONMENT` exported so `uv run --no-sync` executes commands in the named environment without activating it.

## Testing

For a smaller test-only environment, install the `testing` extra:

```bash
uv sync --extra testing
```

This extra contains the test runner, coverage tools, Hydra configuration support, and notebook validation tools. Tests that exercise optional Cellpose, PyTorch, BaSiC, OpenCV, or Squidpy integrations are skipped when those packages are not installed. The `dev` environment installs them so the optional integration paths can run as well.

Run the test suite from the repository root:

```bash
uv run --no-sync pytest
```

Continuous integration will automatically run the tests on pull requests.

When changing dependencies, regenerate and verify the lockfile:

```bash
uv lock
uv lock --check
```

## Automated checks

Install pre-commit as a uv-managed tool and enable the repository hooks:

```bash
uv tool install pre-commit
pre-commit install
pre-commit run --all-files
```

## Documentation contributions

The HTML build is only needed when changing or validating the documentation. Add the `docs` extra to the development environment, then build the local HTML documentation:

```bash
uv sync --extra dev --extra docs
uv run --no-sync python -m sphinx -T --keep-going -b html docs docs/_build/html
```
