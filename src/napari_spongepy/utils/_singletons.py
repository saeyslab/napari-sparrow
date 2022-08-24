from squidpy.im import ImageContainer

_ic = None


def get_ic(*args, **kwargs):
    """Get the reference to the ImageContainer. Access by separate plugins."""
    global _ic
    if not _ic:
        if not kwargs:
            raise ValueError("ImageContainer not initialized.")
        _ic = ImageContainer(*args, **kwargs)
    return _ic
