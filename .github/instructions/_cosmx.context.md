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

### `_read_zarr_image`

- Opens only the `CellComposite` container metadata with Zarr.
- Sorts channel group names for deterministic `c` ordering.
- Calls `_read_cosmx_zarr_levels()` for each channel.
- Stacks matching native levels along `c` with Dask.
- Passes the native arrays to `_build_multiscale_tree()` with `Image2DModel`.

### `_read_zarr_labels`

- Calls `_read_cosmx_zarr_levels()` for the single `CellLabels` store.
- Lazily casts the native label arrays to `uint32`.
- Passes them to `_build_multiscale_tree()` with `Labels2DModel`.

### `_read_cosmx_zarr_levels`

- Uses the OME-Zarr `Reader` and `Multiscales` specification for metadata validation and vendor dataset ordering.
- Does not reconstruct scale paths or scale factors itself.
- Opens the same store with `zarr.open_group()` and creates Dask arrays with `da.from_zarr()` for the dataset paths supplied by `Multiscales.datasets`.
- Validates that every level is 2D.

The direct Zarr pixel step is deliberate for this Windows environment. A focused probe showed that Dask arrays produced through `ome_zarr.ZarrLocation`'s `FSStore` read the synthetic label fixture as zeros, while `zarr.open_group(path, mode="r")` followed by `da.from_zarr(zarr_group[dataset_path])` read the stored value correctly. OME-Zarr remains the source of metadata ordering; direct Zarr remains the source of the lazy pixel arrays.

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
- The vendor table uses one region, `labels_{coordinate_system}`, and cell instances from the vendor cell IDs.
- Sparrow's `ImageLayerManager` and `LabelLayerManager` handle registration and backed output.

## Do Not Reintroduce

- Per-FOV raster discovery or affine estimation.
- TIFF/PNG/JPEG fallback loading.
- Automatic generation of scale factors or downsampled levels.
- Estimation of raster placement from cell centroids.
- Multiple image or labels layers for FOVs when a global vendor store exists.
- In-place modification of vendor `.zattrs` to satisfy SpatialData reader assumptions.

## Dependencies and Private APIs

`ome-zarr>=0.8.4` is declared explicitly because `_cosmx.py` imports `ome_zarr.io.ZarrLocation` and `ome_zarr.reader.Multiscales`/`Reader` directly. The implementation also uses SpatialData's private raster utility functions `_set_transformations` and `compute_coordinates`, matching the pinned `spatialdata==0.4.0` API. If SpatialData is upgraded, retest this adapter and revisit whether `_read_multiscale()` supports raw CosMx metadata directly.

## Validation History

The current modern fixture tests cover native multiscale loading, channel names and order, Dask laziness, identity transforms, labels, table links, backed output, local transcript placement, gene filtering, and rejection of legacy raster inputs.

Verified commands:

```text
uv run pytest src/sparrow/_tests/test_cosmx.py -q
15 passed, 17 warnings

uv run ruff check src/sparrow/io/_cosmx.py src/sparrow/_tests/test_cosmx.py
All checks passed

uv run ruff format --check src/sparrow/io/_cosmx.py src/sparrow/_tests/test_cosmx.py
Both files formatted

uv lock --check
Lockfile is up-to-date and consistent
```

A full repository test run previously exceeded the available two-minute execution window after passing/skipping progress; that timeout was not caused by a reported CosMx failure.
