# CosMx Reader Context

This file is a handoff note for work on `src/sparrow/io/_cosmx.py`.

## Goal

The CosMx reader must support modern global nested OME-Zarr exports while preserving Sparrow behavior:

- Read one global image mosaic from `CellComposite`.
- Read one global labels mosaic from `CellLabels` when `cells_labels=True`.
- Preserve every vendor pyramid level exactly as supplied.
- Keep image and label arrays lazy and Dask-backed.
- Use identity transforms for every layer, and express all coordinates in the mosaic's raster
  pixel space so points, rasters and table centroids overlay.
- Convert the vendor's upward-increasing global pixel y into downward raster rows using the
  mosaic top edge from `fov_positions_file.csv`; use the per-FOV origins to place local
  transcript coordinates.
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
- Non-finite transcript coordinates are rejected. The check runs **inside the Dask graph**, in the
  same partition function that places the coordinates (`_reject_non_finite_coordinates`), so a
  transcript file holding billions of rows is never materialized just to be validated. Both
  coordinate branches go through `map_partitions` for this reason; the error surfaces when
  `add_points_layer` materializes, i.e. still inside the `cosmx()` call.
- Stored transcript layers use Sparrow's canonical `gene`, `x`, and `y` columns and an identity transform.
- `cells_table=True` implies `cells_labels=True` because the table region points to the global labels layer.
- The vendor table uses one region, `labels_{coordinate_system}`, and `_INSTANCE_KEY` holds the
  integer value actually stored in the `CellLabels` mask (see below). The vendor's per-FOV
  `cell_ID` is preserved under `fov_cell_ID`, and the observation index stays the
  `fov`+`cell_ID` string, because the vendor's `cell_ID` restarts at 1 in every FOV.
- Every vendor cell must be unique, and uniqueness is checked on the **paired integer key**, not on
  its `"<fov>_<cell_ID>"` string form. The two are different tests — `"1"` and `"01"` are distinct
  strings that normalize to the same cell — and the paired key is what has to be unique, both
  because it becomes `_INSTANCE_KEY` and because the counts lookup rejects a non-unique index.
- The counts identifier columns are given explicit dtypes when the CSV is streamed. `read_csv`
  infers dtypes independently **per chunk**, so one blank identifier anywhere turns that chunk's
  identifier column into `float64`, whose keys then match nothing and silently drop every cell in
  the chunk. Declaring them turns that into a hard error. Do not remove this.
- Identifier aliases are excluded from the counts gene columns as a set, not just the two chosen
  names, so an export carrying both `cell_ID` and `cell_id` cannot turn the unused alias into a
  fake gene in `var`.
- Counts are streamed into a `scipy` CSR matrix in a single pass, never materialized dense. Rows are
  matched by `pd.Index.get_indexer` on the paired integer key, which is also what `_INSTANCE_KEY`
  holds; the key is built once in `_add_table` and reused, rather than rebuilt as a string per chunk.
- Sparrow's `ImageLayerManager` and `LabelLayerManager` handle registration and backed output.

## Raster Placement: Vendor Global Pixels Are Not Raster Coordinates

The vendor's global pixel coordinate system has **y increasing upward**, while the image and label
mosaics are stored with rows increasing downward. Registering rasters, transcripts and cell centres
all with `Identity()` therefore left the points vertically mirrored against the rasters. Nothing in
the export flags this: the layers simply do not overlay.

The conversion is:

```text
raster column = x_global_px
raster row     = top_global_y - y_global_px
```

where `top_global_y = max(fov_positions_file.csv["y_global_px"])`, because the vendor records each
FOV tile's **top** edge in that y-up system, so the largest origin y is the mosaic's first row.
On the reference export that constant is `116704`, against a level-zero raster of `(120942, 114908)`.

Local transcript coordinates need no flip: local pixels already run left-to-right and top-to-bottom
inside their tile, so each FOV is a plain translation, `row = (top_global_y - origin_y) + y_local_px`.
Note this means the earlier `y_global_px = y_local_px + origin_y` was wrong in sign; the vendor's
own metadata satisfies `y_global_px = origin_y - y_local_px`.

`_load_fov_positions` returns both the origins and `top_global_y`. When the positions file is
missing or has no pixel origin columns, the conversion cannot be computed; the reader logs a
warning and leaves the coordinates in vendor global pixels rather than guessing.

This placement is read from vendor metadata, not estimated from cell centroids — centroids were
used only to verify it (0/1117 FOV-1 cell centres land on a label without the flip, 1117/1117 with
it, and the same constant holds for FOVs spread across the mosaic).

## The Table Region Link Is Real: `CellLabels` Uses Szudzik Pairing

Everywhere else in Sparrow, `_INSTANCE_KEY` is the **integer label value** stored inside the labels
layer, which is what lets a table be joined to its labels layer. The CosMx vendor table *can*
satisfy that contract: the global `CellLabels` mosaic numbers each cell with **Szudzik's elegant
pairing function** applied to `(fov, cell_ID)`:

```text
label = cell_ID*cell_ID + fov          if fov <  cell_ID
label = fov*fov + fov + cell_ID        if fov >= cell_ID
```

`_vendor_label_ids()` implements this, and `_INSTANCE_KEY` is an `int64` mask value. The pairing is
injective, so the key stays unique across the dataset even though `cell_ID` restarts at 1 in every
FOV. Nothing needs to read the mask to build it.

Verification on the reference export: for every FOV tile checked (1, 2, 50, 100, 200, 208, 399,
400), **100%** of the predicted label values were present in that tile; the only extra values were
the 10-79 labels of neighbouring FOVs bleeding over the tile edge. Sampling the mask at each cell
centroid, 98.8% of 9510 cells returned exactly the derived key, the ~1% residual being centroids of
concave cells that fall inside a neighbour.

An earlier version of this note claimed the mask had "no arithmetic relation to `fov`/`cell_ID`",
citing FOV 208 mask values spanning 3 000 - 4 787 385. That measurement read the wrong region of
the mosaic, because it predated the y-flip above. Both branches of the pairing are needed: a naive
`fov*(fov+1) + cell_ID` fits FOV 400 perfectly (all its `cell_ID`s are below 400) but matches only
~3% of FOV 1, where nearly every `cell_ID` exceeds the FOV number.

## Do Not Reintroduce

- Per-FOV raster discovery or affine estimation.
- TIFF/PNG/JPEG fallback loading.
- Automatic generation of scale factors or downsampled levels.
- Estimation of raster placement from cell centroids. (Placement comes from
  `fov_positions_file.csv`; centroids are for verification only.)
- Multiple image or labels layers for FOVs when a global vendor store exists.
- In-place modification of vendor `.zattrs` to satisfy SpatialData reader assumptions.

## Dependencies and Private APIs

`_cosmx.py` no longer imports `ome_zarr` at all, so `ome-zarr` is not declared as a direct dependency of `sparrow`. It stays installed as a transitive dependency of `spatialdata`; nothing in this reader relies on it. See the `_read_cosmx_zarr_levels` notes above for why it was dropped.

The implementation does use SpatialData's private raster utility functions `_set_transformations` and `compute_coordinates`, matching the pinned `spatialdata==0.4.0` API. If SpatialData is upgraded, retest this adapter and revisit whether `_read_multiscale()` supports raw CosMx metadata directly.

## Validation History

The current modern fixture tests cover native multiscale loading, channel names and order, Dask
laziness, identity transforms, labels, table links, backed output, local transcript placement, gene
filtering, rejection of legacy raster inputs, dangling pyramid metadata, duplicate vendor cell keys, half-specified global
transcript coordinates, and the counts/metadata intersection. They also cover the global-y to
raster-row conversion, the matching conversion of table cell centres, the warning path when no FOV
positions are available, the Szudzik pairing behind `_INSTANCE_KEY`, rejection of non-finite
transcript coordinates, rejection of a malformed counts identifier, and the exclusion of a
redundant identifier alias from `var`.

The four "unreadable multiscale metadata" cases and the two "incomplete coordinate columns" cases
are parametrized rather than written as separate tests, matching the house style elsewhere in
`src/sparrow/_tests` (bare tuples, no `ids=`/`pytest.param`, which appear nowhere in this repo).

Verified commands:

```text
uv run --no-sync pytest src/sparrow/_tests/test_cosmx.py -q
29 passed, 6 warnings   # 25 test functions; two of them parametrized

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
