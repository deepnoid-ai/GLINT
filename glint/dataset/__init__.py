from .data_collator import collate_fn  # noqa: F401


def load_datasets(cfg, train, inference, **kwargs):
    if cfg["dataset_type"] == "chest_ct":
        from glint.dataset.chest_ct import load_datasets as load_chest_ct_datasets

        return load_chest_ct_datasets(cfg, train, inference, **kwargs)
    elif cfg["dataset_type"] == "cxr":
        from glint.dataset.cxr import load_datasets as load_cxr_datasets

        return load_cxr_datasets(cfg, train, inference, **kwargs)
    else:
        raise NotImplementedError
