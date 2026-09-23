---
description: "Use when writing, editing, or reviewing Python code in napari-sparrow (sparrow). Covers commenting conventions, type hints, docstrings, naming, logging, error handling, and SpatialData patterns."
applyTo: "**/*.py"
---

# napari-sparrow Python Coding Instructions

## Commenting & Annotation (Highest Priority)

If you are trying to adjust Jupyter Notebooks in the repository, check if the adjustments actually stuck. If they did not stick, either find a way to make them stick or show the code changes in the Copilot chat.

Every non-obvious line of code **must** have at least one comment line directly above it explaining what the code is doing. This is the single most important rule in this codebase. For obvious lines of code, a comment line is optional or not necessary. For example, a simple assignment of a variable to a constant does not require a comment line, but a complex operation or transformation should be annotated.

```python
# Good — every non-trivial statement is annotated
# Convert labels layer to a list if an iterable was passed (but not a plain string)
labels_layer = (
    list(labels_layer)
    if isinstance(labels_layer, Iterable) and not isinstance(labels_layer, str)
    else [labels_layer]
)

# Select only the channels that are present in both the image and the provided list
channels = [c for c in image.dims if c in requested_channels]

# Persist dask graph to avoid recomputation during downstream operations
image = image.persist()
```

For complex operations, use **multiple comment lines** to explain the full intent, each step, and any non-obvious choices:

```python
# Squeeze single-channel dimension so that downstream 2D filters work correctly.
# Only the c-dimension (axis 0) is removed; spatial axes are left untouched.
# If the channel count is > 1 we raise because the function only handles c==1.
if image.shape[0] == 1:
    image = da.squeeze(image, axis=0)
else:
    raise ValueError("_apply_min_max_filtering only accepts c dimension equal to 1.")
```

Inline (end-of-line) comments are **not** a substitute for annotation lines above. Write the comment above the code it describes, not beside it.

One-liner utility functions or properties may use a single-line docstring instead of an above-comment if the name is self-explanatory.

## Imports

Always place `from __future__ import annotations` as the first import so that forward references in type hints are resolved lazily:

```python
from __future__ import annotations

from typing import Any
```

## Type Hints

- Use the `|` union syntax (`str | None`) instead of `typing.Union`
- Use built-in generic types (`list[str]`, `dict[str, int]`) instead of `List`, `Dict`
- Annotate all function parameters and return types

```python
def segment(
    sdata: SpatialData,
    img_layer: str | None = None,
    labels_layer: str | list[str] | None = None,
    overwrite: bool = False,
) -> SpatialData:
```

## Docstrings

Use **NumPy-style** docstrings for all public functions and methods. Include `Parameters`, `Returns`, `Raises`, and `Examples` sections where relevant:

```python
def enhance_contrast(
    sdata: SpatialData,
    img_layer: str | None = None,
    contrast_clip: float = 2.0,
) -> SpatialData:
    """Enhance contrast of an image layer in a SpatialData object.

    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to each
    channel of the specified image layer.

    Parameters
    ----------
    sdata
        Spatial data object containing the image to process.
    img_layer
        The image layer to enhance. Uses the last added layer if ``None``.
    contrast_clip
        Clip limit for CLAHE. Higher values increase contrast more aggressively.

    Returns
    -------
    The updated ``sdata`` object with the contrast-enhanced image added.

    Raises
    ------
    ValueError
        If ``img_layer`` is not found in ``sdata``.
    """
```

Private helpers (`_prefixed`) may use shorter one-liner or abbreviated docstrings.

## Naming Conventions

- **Functions & variables**: `snake_case`
- **Classes**: `PascalCase`
- **Private functions / methods**: prefix with `_` (e.g. `_apply_filter`)
- **Module-level constants / keys**: ALL_CAPS or `_SCREAMING_SNAKE` for private constants (e.g. `_INSTANCE_KEY`)
- Parameter names must be descriptive; avoid single letters except for well-known math variables

## Logging

Every module that performs meaningful computation defines a module-level logger using the project utility:

```python
# Set up a module-scoped logger using the project helper so output is consistent
log = get_pylogger(__name__)
```

Use `log.info` for major pipeline steps, `log.warning` for recoverable situations where the user should take note, and `log.debug` for fine-grained diagnostic output. Never use `print()`.

## Error Handling

- Use `ValueError` with a descriptive f-string message for invalid arguments or states
- Use `try/except ImportError` with a `log.warning` for optional dependencies; never hard-fail on missing optional packages
- Validate inputs early (at function entry) and raise before any expensive computation starts

```python
# Validate that the filter size list matches the number of image channels
if isinstance(size_min_max_filter, list) and len(size_min_max_filter) != num_channels:
    raise ValueError(
        f"'size_min_max_filter' has {len(size_min_max_filter)} entries "
        f"but the image has {num_channels} channels."
    )
```

## SpatialData Conventions

- All public API functions accept a `SpatialData` object as the first argument and return the (updated) `SpatialData` object
- Results are added to `sdata` in-place and the same object is returned to allow chaining
- Use `overwrite: bool = False` as the last keyword argument to control whether an existing layer may be replaced

## Dask & Array Operations

- Keep image data as `dask` arrays to support lazy, out-of-core computation on large spatial datasets
- Rechunk or persist only when necessary, and annotate **why** with a comment
- Avoid `.compute()` unless the result must be materialised for a non-dask operation; document the reason in a comment

## Tests

- Use `pytest` with function-scoped fixtures
- Guard optional-dependency tests with `@pytest.mark.skipif(not importlib.util.find_spec(...), reason="requires ...")`
- Test file names mirror source file names: `image/min_max.py` → `_tests/test_min_max.py`
