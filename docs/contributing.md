# Development

## Setting up a development environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and clone the GitHub repository:

```bash
git clone https://github.com/saeyslab/napari-sparrow.git
cd napari-sparrow
```

The project requires Python 3.11.15. Create the virtual environment and install the development dependencies:

```bash
uv python install 3.11.15
uv venv --python 3.11.15
uv sync --extra dev
```

The `dev` extra includes the testing, notebook, and `rioxarray` dependencies. Use `uv run` to execute commands in the project environment without activating it.

## Testing

To run unit tests, run the following from the root of the project:

```bash
uv run --no-sync pytest
```

Continuous integration will automatically run the tests on all pull requests.

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
