# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SPArrOW is a library for spatial transcriptomics/proteomics analysis, built on top of
[`spatialdata`](https://github.com/scverse/spatialdata). The distribution name is `sparrow`
(the repo is still named `napari-sparrow`); the import name is `sparrow`.

Python 3.11 only (`requires-python = "==3.11.*"`). Dependencies are managed with `uv`
(`uv.lock` is committed). Several pins are load-bearing and should not be bumped casually:
`spatialdata==0.4.0`, `numpy<2`, `pyarrow<=21.0.0` (newer pyarrow breaks spatialdata),
`omegaconf==2.3.0`.

## Commands

```bash
# Environments (uv creates/uses .venv; `uv run` needs no activation)
uv sync                          # core only
uv sync --extra testing          # test runner + hydra + notebook validation
uv sync --extra dev              # testing + tutorials (plugin, notebook, tiling, riox, dashboard)
uv sync --extra dev --extra docs # + sphinx

# Tests (testpaths = src/sparrow/_tests)
uv run --no-sync pytest
uv run --no-sync pytest src/sparrow/_tests/test_cosmx.py -q           # one file
uv run --no-sync pytest src/sparrow/_tests/test_cosmx.py -k test_name # one test
uv run --no-sync pytest --cov=sparrow --cov-report=xml:coverage.xml   # as CI runs it

# Lint / format (ruff, line-length 120, numpy docstring convention)
uv run ruff check <paths>
uv run ruff format <paths>
uv run ruff format --check <paths>

# Pre-commit (prettier + ruff + hygiene hooks)
uv tool install pre-commit && pre-commit install
pre-commit run --all-files

# Lockfile — regenerate and verify whenever dependencies change
uv lock && uv lock --check

# Docs
uv run --no-sync python -m sphinx -T --keep-going -b html docs docs/_build/html
```

The full suite downloads example datasets via `pooch` on first run and can exceed a two-minute
timeout; prefer targeting a single test file while iterating.

Tests for optional integrations (cellpose, torch, basicpy, opencv, squidpy, sklearn, rasterio)
are guarded with `@pytest.mark.skipif(not importlib.util.find_spec(...))` and silently skip in a
`testing`-only environment. `uv sync --extra dev` is what actually exercises them.

Tutorial notebooks live in the `docs/tutorials` Git submodule and are stored in Git LFS. They are
only needed for `test_notebooks.py` and the docs build:
`git submodule update --init --recursive && git -C docs/tutorials lfs pull`.

## Architecture

### Everything is a SpatialData layer transformation

The whole library is one shape of function, applied over and over:

```python
def some_operation(sdata, <input>_layer, ..., output_layer, overwrite: bool = False) -> SpatialData
```

`sdata` is always the first argument; inputs and outputs are referenced by **layer name strings**,
not by passing arrays around; the result is added to `sdata` and the same object is returned so
calls chain. `overwrite` is the last keyword argument and guards replacing an existing layer.
Follow this contract for any new public function — the CLI, the napari widgets, and the notebooks
all depend on it.

`sdata` may be **backed** by a zarr store or purely in-memory, and this distinction drives real
behavior: in-memory arrays get `.persist()`ed to avoid recomputation, while backed layers are
written through `sdata.write_element()` and then re-read so the in-memory handle points at disk.
Overwriting a backed layer goes through `sparrow/utils/_io.py::_incremental_io_on_disk`, which
writes to a temporary UUID-suffixed element first because zarr cannot safely overwrite an element
that is still being read from.

### Layer managers

That backed/in-memory bookkeeping is centralized in manager classes rather than repeated:

- [image/_manager.py](src/sparrow/image/_manager.py) — `LayerManager` (ABC), `ImageLayerManager`, `LabelLayerManager`
- [shape/_manager.py](src/sparrow/shape/_manager.py) — `ShapesLayerManager` (also holds mask to polygon vectorization)
- [table/_manager.py](src/sparrow/table/_manager.py) — `TableLayerManager`

Public entry points are thin wrappers: `im.add_image_layer`, `im.add_labels_layer`,
`sh.add_shapes_layer`, `tb.add_table_layer`, `pt.add_points_layer`. **New code that writes a layer
should go through these, not assign to `sdata[...]` directly.**

### Public API and the `.pyi` stubs

`import sparrow as sp` exposes namespace aliases, imported in a fixed order in
[`__init__.py`](src/sparrow/__init__.py) to avoid circular imports:

| alias                     | module           | contents                                                                                         |
| ------------------------- | ---------------- | ------------------------------------------------------------------------------------------------ |
| `sp.io`                   | `sparrow.io`     | vendor readers: `cosmx`, `merscope`, `xenium`, `visium_hd`, `read_*_transcripts`, `create_sdata` |
| `sp.im`                   | `sparrow.image`  | image/labels ops, segmentation                                                                   |
| `sp.sh`                   | `sparrow.shape`  | polygon layers                                                                                   |
| `sp.pt`                   | `sparrow.points` | transcript/points layers                                                                         |
| `sp.tb`                   | `sparrow.table`  | `AnnData` table layers                                                                           |
| `sp.pl`                   | `sparrow.plot`   | plotting                                                                                         |
| `sp.utils`, `sp.datasets` |                  | logging, aggregation, queries; pooch-backed example data                                         |

Every subpackage keeps an `__init__.pyi` stub listing its public names. `io`, `points`, `shape`,
`utils`, and `image.segmentation*` are **lazily loaded** via `lazy.attach_stub`, so for those the
`.pyi` _is_ the API definition — a function not listed there is not importable. `image`, `table`,
`plot`, and `datasets` import eagerly in `__init__.py` and mirror it in `.pyi`. When adding a
public function, update the `.pyi` (and `__init__.py` for the eager ones) **and** `docs/api.md`,
which drives the autosummary API reference.

### Dask-first, blockwise processing

Image and label data stay as lazy dask arrays so out-of-core processing works on whole-slide data.
`im.map_image` ([image/_map.py](src/sparrow/image/_map.py)) and `im.map_labels`
([image/segmentation/_map.py](src/sparrow/image/segmentation/_map.py)) are the shared machinery:
they handle per-channel/per-z-slice dispatch of `func` (a `Mapping` keyed by channel/z-slice
selects different callables or kwargs), rechunking, cropping to `crd` in a given
`to_coordinate_system`, and choose `dask.array.map_overlap` when `depth` is set versus
`map_blocks` otherwise. Most image operations (`enhance_contrast`, `min_max_filtering`,
`gaussian_filtering`, `normalize`, ...) are thin `func`s routed through `map_image`.

Segmentation follows the same idea: `im.segment` takes a `model` callable of
`(z, y, x, c) -> (z, y, x, c)` labels and handles chunked execution plus cross-chunk label
reconciliation (IoU-based relabeling at chunk borders, controlled by `iou`, `iou_depth`,
`iou_threshold`, `trim`). Concrete models live in
[image/segmentation/segmentation_models/](src/sparrow/image/segmentation/segmentation_models/)
and are exposed as `*_callable` factories (`im.cellpose_callable`, `im.baysor_callable`).

### Tables link back to labels via reserved keys

[utils/_keys.py](src/sparrow/utils/_keys.py) defines the private column names that tie an
`AnnData` table layer to a labels layer: `_REGION_KEY = "fov_labels"` (which labels layer a row
belongs to) and `_INSTANCE_KEY = "cell_ID"` (which cell). `TableLayerManager.add_table` asserts
both exist in `adata.obs` before `TableModel.parse`. `tb.allocate` (transcripts to counts) and
`tb.allocate_intensity` (channel intensities to counts) produce these tables; `append=True` adds
another labels layer as an extra region to an existing table. Use the constants, never the literal
strings.

### Coordinate systems

Layers carry `spatialdata` transformations into named coordinate systems (default `"global"`).
Sparrow generally supports `Identity`, `Translation`, and `Sequence` of translations — see
`sparrow/image/_image.py::_get_translation`, which raises on anything else. Functions that take
`crd` also take `to_coordinate_system` to say which system the crop is expressed in.

### Two front ends over the same library

- **Hydra CLI** — `sparrow` console script to [single.py](src/sparrow/single.py) to
  [pipeline.py](src/sparrow/pipeline.py) (`SparrowPipeline`: load, clean, segment, allocate,
  annotate, visualize). Defaults live in [src/sparrow/configs/](src/sparrow/configs/) (packaged);
  runnable per-dataset examples live in the top-level [configs/](configs/) folder, which users copy
  and point at with `hydra.searchpath`.
- **napari plugin** — [napari.yaml](src/sparrow/napari.yaml) registers widgets in
  [widgets/](src/sparrow/widgets/); only the Wizard widget is currently enabled. Widgets are
  excluded from coverage.

### Example data

[datasets/registry.py](src/sparrow/datasets/registry.py) is a `pooch` registry (SHA256-pinned,
hosted on VIB object storage). `sp.datasets.*_example()` helpers download and open ready-made
`SpatialData` objects; test fixtures in
[_tests/conftest.py](src/sparrow/_tests/conftest.py) build on them.

### Test layout

`src/sparrow/_tests/` mirrors the source tree (`image/_filters.py` to `_tests/test_image/test_filters.py`).
Shared fixtures in `conftest.py` provide backed and non-backed variants of the same dataset
(`sdata_transcripts` vs `sdata_transcripts_no_backed`) — use both when touching layer-writing code,
since the backed and in-memory paths differ.

## Code conventions

Enforced by review rather than by the linter. The authoritative version is
[.github/instructions/sparrow-python.instructions.md](.github/instructions/sparrow-python.instructions.md);
read it before writing Python here. Summary:

- **Comment density is the house style, and the highest-priority rule.** Every non-obvious
  statement gets at least one comment line _directly above_ it explaining what the code does;
  complex operations get several lines covering intent, each step, and any non-obvious choice.
  Obvious lines need none. Self-explanatory one-liner utilities may use a single-line docstring instead.
- `from __future__ import annotations` as the first import in every module.
- `str | None` unions and builtin generics (`list[str]`), never `typing.Union`/`List`. Annotate all
  parameters and return types.
- NumPy-style docstrings on public functions (`Parameters`, `Returns`, `Raises`, `Examples`).
  Parameter descriptions omit the type line (it's in the signature). Private `_helpers` may use
  short docstrings.
- Naming: `snake_case` functions/variables, `PascalCase` classes, `_` prefix for private,
  ALL_CAPS module constants. Descriptive parameter names, no single letters.
- Logging via `log = get_pylogger(__name__)` from `sparrow.utils`; `log.info` for pipeline steps,
  `log.warning` for recoverable situations, `log.debug` for diagnostics. Never `print()`.
- Validate arguments at function entry and `raise ValueError(f"...")` before any expensive
  computation. Optional dependencies are probed with `importlib.util.find_spec` and degrade with a
  `log.warning`/`log.info` fallback rather than hard-failing.
- Keep image data as dask arrays; avoid `.compute()` unless materialization is genuinely required,
  and comment why. Rechunk or persist only when necessary, with a comment explaining the reason.
- When editing Jupyter notebooks, verify the edits actually landed in the file before reporting.

## Vendor reader notes

`sparrow.io` readers adapt vendor exports into SpatialData layers. The CosMx reader
([io/_cosmx.py](src/sparrow/io/_cosmx.py)) is the most involved one: it reads global nested
OME-Zarr mosaics directly rather than delegating to `spatialdata_io.cosmx`, and it depends on
`ome-zarr` plus private `spatialdata` internals (`_set_transformations`, `compute_coordinates`)
that match the pinned `spatialdata==0.4.0`. Before changing it, read
[.github/instructions/_cosmx.context.md](.github/instructions/_cosmx.context.md) — it records the
vendor layout, why `read_zarr()`/`_read_multiscale()` are not usable entry points, and an explicit
"do not reintroduce" list. If `spatialdata` is ever upgraded, that reader is the first thing to
retest.

## CI

[.github/workflows/run_tests.yml](.github/workflows/run_tests.yml) runs pytest on Ubuntu /
Python 3.11 in a conda env, but only for pushes/PRs touching `src/sparrow/**`, the workflow itself,
`.gitmodules`, or `.readthedocs.yaml`. It fails loudly on unresolved Git LFS pointers in
`docs/tutorials`. [.github/workflows/docs.yml](.github/workflows/docs.yml) builds the Sphinx docs
on every PR.
