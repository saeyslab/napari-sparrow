def parse_subset(subset):
    """
    e.g $ sparrow subset=\'0:100,0:100\'
    >>> parse_subset('0:100,0:100')
    return left corner and size ([0, 0], [100, 100])

    """
    left_corner = []
    size = []
    for x in subset.split(","):
        left_corner.append(int(x.split(":")[0]))
        size.append(int(x.split(":")[1]) - left_corner[-1])
    return left_corner, size


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
