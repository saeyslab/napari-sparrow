def parse_subset(subset):
    """
    e.g $ python src/segment.py subset=\'0:100,0:100\'
    >>> parse_subset('0:100,0:100')
    (slice(0, 100, 1), slice(0, 100, 1))
    """
    return tuple(
        slice(int(x.split(":")[0]), int(x.split(":")[1]), 1) for x in subset.split(",")
    )
