import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_align_masks": ["align_labels_layers"],
        "_expand_masks": ["expand_labels_layer"],
        "_segmentation": ["segment"],
    },
)
