def parse_subset(subset):
    """
    e.g $ python src/segment.py subset=\'0:100,0:100\'
    >>> parse_subset('0:100,0:100')
    (slice(0, 100, 1), slice(0, 100, 1))
    """
    return tuple(
        slice(int(x.split(":")[0]), int(x.split(":")[1]), 1) for x in subset.split(",")
    )


def ic_to_da(
    ic, label="image", drop_dims=["z", "channels"], reduce_z=None, reduce_c=None
):
    """
    Convert ImageContainer to dask array.
    ImageContainer defaults to (x, y, z, channels (c if using xarray format)), most of the time we need just (x, y)
    The c channel will be named c:0 after segmentation.
    """
    if reduce_z or reduce_c:
        # TODO solve c:0 output when doing .isel(z=reduce_z, c=reduce_c)
        return ic[label].isel({"z": reduce_z, "c:0": 0}).data
    else:
        return ic[label].squeeze(dim=drop_dims).data
