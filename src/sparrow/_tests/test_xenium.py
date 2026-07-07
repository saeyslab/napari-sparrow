from __future__ import annotations

import zarr

from sparrow.io._xenium import _zipstore_mode_compat


def test_zipstore_mode_compat_translates_read_only(monkeypatch):
    # Track how the dummy ZipStore is called so we can assert on the translated mode.
    calls: list[dict[str, object]] = []

    # Provide a lightweight stand-in for the installed ZipStore implementation.
    class DummyZipStore:
        def __init__(self, *args, mode: str = "a", **kwargs):
            # Instead of doing real ZipStore work, we appends the received arguments to calls
            calls.append({"args": args, "mode": mode, "kwargs": kwargs})

    # Replace the real ZipStore with the dummy base class as the compatibility shim wraps whatever ZipStore exists at runtime. 
    # By swapping in a fake first, the test avoids touching the real zarr implementation and can inspect exactly what gets passed through.
    monkeypatch.setattr(zarr.storage, "ZipStore", DummyZipStore)

    # Enter the compatibility shim and call ZipStore using the read_only argument which is what spatialdata-io 0.6.0 would call
    with _zipstore_mode_compat("r"):
        zarr.storage.ZipStore("cells.zip", read_only=True)

    # Confirm that the legacy argument was translated into the requested mode.
    assert calls == [{"args": ("cells.zip",), "mode": "r", "kwargs": {}}]
