# Development

## Setting up a development environment

First clone the GitHub repo and set it as the current directory:

```bash
git clone https://github.com/saeyslab/napari-sparrow.git
cd napari-sparrow
```

Create a conda virtual environment as explained [here](./installation.md) and install `SPArrOW`.

```bash
conda activate napari-sparrow
pip install -e '.[testing,docs]'
```

This development environment is supported for:

- CentOS
- Ubuntu
- MacOS with an M1/M2 Pro
- Windows 11

## Testing

To run unit tests, run the following from the root of the project:

```bash
pytest
```

Continuous integration will automatically run the tests on all pull requests.

### Type testing

Do a type test:

```
mypy --ignore-missing-imports src/
```

## Automated commit checks

Install a pre-commit hook to run all configured checks in `.pre-commit-config.yaml`:

```
pre-commit install
pre-commit run -a
```
