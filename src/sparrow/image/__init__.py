import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_apply": ["apply"],
        "_combine": ["combine"],
        "_contrast": ["enhance_contrast"],
        "_minmax": ["min_max_filtering"],
        "_tiling": ["tiling_correction"],
        "_transcripts": ["transcript_density"],
        "segmentation": ["segment", "align_labels_layers", "expand_labels_layer"],
        "_image": ["_add_image_layer", "_add_label_layer"],
    },
)
