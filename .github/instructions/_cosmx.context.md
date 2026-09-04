# CosMx Reader Context

This file is a handoff note for work on `src/sparrow/io/_cosmx.py`.

## Goal

The CosMx reader must support modern global nested OME-Zarr exports while preserving Sparrow behavior:

- Read one global image mosaic from `CellComposite`.
- Read one global labels mosaic from `CellLabels` when `cells_labels=True`.
- Preserve every vendor pyramid level exactly as supplied.
- Keep image and label arrays lazy and Dask-backed.
- Use identity transforms for global rasters and global transcript coordinates.
- Use FOV origin translations only when transcripts have local coordinates.
- Keep labels and the optional vendor table linked through Sparrow table metadata.
- Do not restore the legacy TIFF/PNG fallback or generate replacement scale levels.

## Observed Vendor Layout

The inspected dataset is:

`D:\CosMx_test_datasets\CosMx_reader_test_dataset`

Its relevant structure is:

- `CellComposite/`
  - `CD45/`, `CD68/`, `DNA/`, `Membrane/`, `PanCK/`
  - Each channel directory is an OME-Zarr multiscale group with arrays `0` through `7`.
- `CellLabels/`
  - One OME-Zarr multiscale group with arrays `0` through `7`.
- `S0_tx_file.csv`
- `S0_fov_positions_file.csv`
- `S0_exprMat_file.csv`
- `S0_metadata_file.csv`

The real level-zero arrays are approximately `(120942, 114908)` with native chunks `(8192, 8192)`. Image channels are `uint16`; labels are `uint32`. The vendor metadata records physical scales beginning at `0.1203` and doubling at each level. The global raster coordinate convention in Sparrow is pixel coordinates, so the level-zero grid is represented by pixel-center coordinates and lower levels are positioned relative to it.

The small test fixture uses the same nested structure, but two levels and three channels. Its OME-Zarr metadata intentionally has per-dataset scale transformations and no top-level `coordinateTransformations` entry.

## Why `read_zarr()` Is Not the Entry Point

`spatialdata.read_zarr()` reads a SpatialData container. It expects reserved groups such as:

- `images/<layer>`
- `labels/<layer>`
- `points/<layer>`
- `tables/<layer>`

A CosMx export is not already a SpatialData container. `CellComposite` is a vendor Zarr group whose children are independent channel OME-Zarr stores, and `CellLabels` is a raw vendor OME-Zarr store. Therefore the CosMx reader must discover those groups, read their native arrays, assemble an image channel dimension, and then register the resulting SpatialData element with Sparrow's managers.

`spatialdata_io.cosmx` is not reusable for this path. The installed implementation is built around per-FOV TIFF/PNG/JPEG files, flips each raster, estimates per-FOV affine transforms from table centroids, and creates FOV-specific layers. Those assumptions are wrong for global CosMx OME-Zarr mosaics.

## `_read_multiscale()` Compatibility Detail

The installed `spatialdata._io.io_raster._read_multiscale(store, raster_type)` is a useful generic reader, but its implementation in the pinned `spatialdata==0.4.0` has a metadata assumption that raw CosMx groups do not satisfy.

Its relevant sequence is:

1. Open the store and construct an `ome_zarr.reader.Reader`.
2. Find a node containing an OME-Zarr `Multiscales` specification.
3. Load the node's `multiscales` metadata.
4. Read `multiscales[0]["coordinateTransformations"]`.
5. Convert those top-level transformations into SpatialData transforms.
6. Load the arrays and build a `DataArray` or `DataTree`.

There are two different metadata locations that are easy to confuse:

```text
multiscales[0]["coordinateTransformations"]
multiscales[0]["datasets"][level]["coordinateTransformations"]
```

The CosMx stores inspected here have the second form: every dataset has a scale transformation such as `[0.1203, 0.1203]`, `[0.2406, 0.2406]`, and so on. They do not have the first, top-level form. Consequently, the pinned implementation fails at step 4 with:

```text
KeyError: 'coordinateTransformations'
```

This happens before `_read_multiscale()` returns any arrays. It is a metadata-schema mismatch, not a missing image or a failed pixel read. Adding an empty field directly to the vendor store would mutate input metadata and is unsafe, especially for concurrent readers, so the adapter does not patch vendor files in place.

There is a second classification issue for labels: `_read_multiscale(..., raster_type="labels")` selects nodes carrying the OME-Zarr `Label` marker (`image-label` metadata). Raw CosMx `CellLabels` metadata is not guaranteed to carry that marker, even though its arrays are integer label masks. Therefore blindly delegating labels to `_read_multiscale()` is not reliable either.

## Current Raster Adapter

The raster-specific code in `_cosmx.py` is intentionally limited to vendor composition:

### `_process_multiscale_raster`

One function handles both raster kinds, selected by its `kind` argument.

For `kind="image"`:

- Opens only the `CellComposite` container metadata with Zarr.
- Sorts channel group names for deterministic `c` ordering.
- Calls `_read_cosmx_zarr_levels()` for each channel.
- Validates that every channel exposes the same number of pyramid levels.
- Stacks matching native levels along `c` with Dask.
- Passes the native arrays to `_build_multiscale_tree()` with `Image2DModel`.

For `kind="labels"`:

- Calls `_read_cosmx_zarr_levels()` for the single `CellLabels` store.
- Lazily casts the native label arrays to `uint32`.
- Passes them to `_build_multiscale_tree()` with `Labels2DModel`.

### `_read_cosmx_zarr_levels`

- Opens the store once with `zarr.open_group()`, which is the source of both the metadata and the arrays.
- Reads the vendor dataset ordering from that group's own `multiscales` attribute; it does not reconstruct scale paths or scale factors itself.
- Requires exactly one `multiscales` entry, a non-empty `datasets` list, and every named level to be present in the store.
- Creates Dask arrays with `da.from_zarr()` for those dataset paths.
- Validates that every level is 2D.

Direct Zarr access is deliberate, for two separate reasons.

For **pixels**, a focused probe on this Windows environment showed that Dask arrays produced through `ome_zarr.ZarrLocation`'s `FSStore` read the synthetic label fixture as zeros, while `zarr.open_group(path, mode="r")` followed by `da.from_zarr(zarr_group[dataset_path])` read the stored value correctly.

For **metadata ordering**, `ome_zarr.reader.Reader` was used until it was measured against reading `group.attrs["multiscales"]` directly. Both return byte-identical levels on the reference export: same ordering, shapes, dtypes, chunking and pixel content across all six stores. But `ome_zarr.reader.Multiscales.__init__` eagerly builds and then discards a Dask array for every level, which on a full read of that export produced 48 spurious `ignoring keyword argument 'read_only'` warnings (8 per store) plus one logged `Failed to parse metadata` traceback per channel, because the vendor's `omero` metadata names channel colours in words (`"red"`, `"blue"`) where OME-Zarr expects hex. Reading the attribute directly produces zero of either and is ~2.8x faster, though both are well under a tenth of a second.

The trade-off accepted here: `Reader` walks child groups, so it would reject a store carrying a second multiscale pyramid in a subgroup. Reading the attribute only sees the pyramid the group itself declares. No CosMx store observed so far has such a subgroup.

### `_build_multiscale_tree`

- Optionally rechunks each existing native array without changing its shape or values.
- Parses each level using the appropriate SpatialData raster model.
- Builds the standard `scale0`, `scale1`, ... `DataTree`.
- Uses SpatialData's `_set_transformations()` with `Identity()` at scale zero; SpatialData derives lower-level scale transforms from native array shapes.
- Uses SpatialData's `compute_coordinates()` to express lower-level pixel-center coordinates in the level-zero pixel grid.

No downsampling, rescaling, or replacement pyramid generation occurs.

## Non-Raster CosMx Behavior

Keep the following behavior stable when changing the reader:

- `path` and `to_coordinate_system` can be scalars or lists; lengths must match and coordinate-system names must be unique.
- `dataset_id` selects related transcript/FOV/table files. When omitted, it is inferred from the transcript filename.
- Parquet transcripts are preferred over CSV transcripts.
- `keep_gene_names` filters transcripts before coordinate work and filters the optional vendor table. A scalar gene name is rejected; string values are interpreted as panel paths.
- Global transcript columns (`x_global_px`, `y_global_px`) take precedence over local columns.
- Local transcript coordinates use `fov_positions_file.csv` origins when there are usable origins. Multiple local FOVs without complete origins are rejected.
- Non-numeric and non-finite transcript coordinates are rejected before layer creation.
- Stored transcript layers use Sparrow's canonical `gene`, `x`, and `y` columns and an identity transform.
- `cells_table=True` implies `cells_labels=True` because the table region points to the global labels layer.
- The vendor table uses one region, `labels_{coordinate_system}`, and cell instances keyed by a
  dataset-global `fov`+`cell_ID` string, because the vendor's `cell_ID` restarts at 1 in every FOV.
- The vendor `fov`+`cell_ID` key must be unique in the metadata file; a duplicate is rejected.
- Counts are streamed into a `scipy` CSR matrix in a single pass, never materialized dense.
- Sparrow's `ImageLayerManager` and `LabelLayerManager` handle registration and backed output.

## Known Limitation: The Table Region Link Is Nominal

Everywhere else in Sparrow, `_INSTANCE_KEY` is the **integer label value** stored inside the labels
layer, which is what lets a table be joined to its labels layer. The CosMx vendor table cannot
satisfy that contract, and this is accepted rather than worked around.

The vendor's global `CellLabels` mosaic is numbered with its own dataset-wide integer IDs that have
no arithmetic relation to `fov`/`cell_ID`. Measured on the reference export
(`D:\CosMx_test_datasets\CosMx_reader_test_dataset`), FOV 208 holds 1782 cells with `cell_ID`
running 1-1782, while the mask values inside that FOV's tile span roughly 3 000 - 4 787 385. The
export ships no mapping between the two numbering schemes: `S0_metadata_file.csv` carries `cell_ID`
(per-FOV `int`) and `cell_id` (`c_<slide>_<fov>_<cell_ID>`, globally unique but still not the mask
value), and `S0-polygons.csv` carries the same per-FOV `cellID`.

Consequences to keep in mind:

- `table_{cs}` declares `region=["labels_{cs}"]`, but that link cannot actually be resolved.
- `_INSTANCE_KEY` is a `str` here, not an `int`. Consumers that resolve a table against a labels
  layer through it cast with `astype(int)` (`sparrow/shape/_manager.py::filter_shapes`,
  `sparrow/plot/_plot.py`) and will not work against this table.
- `tb.allocate(..., append=True)` onto this table would mix `str` and `int` instance keys.

Reconstructing a true integer instance key would mean sampling the label mosaic per cell. That was
considered and deliberately not done. If it is ever revisited, note that on the probe above only
about 1360 of 1782 cell centroids landed on a nonzero mask value, and some distinct `cell_ID`s
sampled to the same mask label, so centroid sampling alone is not a faithful mapping.

## Do Not Reintroduce

- Per-FOV raster discovery or affine estimation.
- TIFF/PNG/JPEG fallback loading.
- Automatic generation of scale factors or downsampled levels.
- Estimation of raster placement from cell centroids.
- Multiple image or labels layers for FOVs when a global vendor store exists.
- In-place modification of vendor `.zattrs` to satisfy SpatialData reader assumptions.

## Dependencies and Private APIs

`_cosmx.py` no longer imports `ome_zarr` at all, so `ome-zarr` is not declared as a direct dependency of `sparrow`. It stays installed as a transitive dependency of `spatialdata`; nothing in this reader relies on it. See the `_read_cosmx_zarr_levels` notes above for why it was dropped.

The implementation does use SpatialData's private raster utility functions `_set_transformations` and `compute_coordinates`, matching the pinned `spatialdata==0.4.0` API. If SpatialData is upgraded, retest this adapter and revisit whether `_read_multiscale()` supports raw CosMx metadata directly.

## Validation History

The current modern fixture tests cover native multiscale loading, channel names and order, Dask
laziness, identity transforms, labels, table links, backed output, local transcript placement, gene
filtering, rejection of legacy raster inputs, dangling pyramid metadata, duplicate vendor cell keys, half-specified global
transcript coordinates, and the counts/metadata intersection.

Verified commands:

```text
uv run --no-sync pytest src/sparrow/_tests/test_cosmx.py -q
25 passed, 6 warnings

uv run --no-sync ruff check src/sparrow/io/_cosmx.py src/sparrow/_tests/test_cosmx.py
All checks passed

uv run --no-sync ruff format --check src/sparrow/io/_cosmx.py src/sparrow/_tests/test_cosmx.py
2 files already formatted
```

The counts streaming path was additionally checked against a brute-force dense reference (matrix
contents, the `seen` mask, gene-column ordering under a whitelist, the zero-overlap case, and cells
whose counts are all zero).

A full repository test run previously exceeded the available two-minute execution window after
passing/skipping progress; that timeout was not caused by a reported CosMx failure.
