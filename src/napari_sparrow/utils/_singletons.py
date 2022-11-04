from squidpy.im import ImageContainer

from napari_sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

_ic = None


def get_ic(*args, **kwargs):
    """Get the reference to the ImageContainer. Access by separate plugins."""
    global _ic
    if not _ic:
        if not kwargs:
            raise ValueError("ImageContainer not initialized.")
        _ic = ImageContainer(*args, **kwargs)
    else:
        if args or kwargs:
            try:
                if "layer" in kwargs.keys() and kwargs["layer"] in _ic:
                    log.debug(
                        f"Trying to make an ImageContainer with layer `{kwargs['layer']}` that already exists. Ignoring and returning existing ImageContainer."
                    )
                    log.debug(_ic)
                    return _ic
                else:
                    log.debug(
                        f"Adding image from layer `{kwargs['layer']}` to existing ImageContainer."
                    )
                    _ic.add_img(*args, **kwargs)
                    log.debug(_ic)
                    return _ic
            except Exception:
                raise ValueError("ImageContainer already initialized.")
    return _ic
