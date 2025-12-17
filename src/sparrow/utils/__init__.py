import lazy_loader as lazy

LOAD = "raw_image"
IMAGE = "image"
CLEAN = "cleaned"
SEGMENT = "segment"
ALLOCATION = "allocation"

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
