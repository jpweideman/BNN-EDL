"""DataLoader creation from named loader configurations."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split

from src.builders.dataset_builder import DatasetBuilder
from src.builders.transform_builder import TransformBuilder

DATA_DIR = str(Path(__file__).parent.parent.parent / "data")


def create_loaders(datasets_cfg):
    """
    Create one DataLoader per named loader config.

    Args:
        datasets_cfg: DictConfig mapping loader name to loader config.
            Each entry must have: name, source, batch_size, num_workers.
            Optional: shuffle (default False), transforms,
                      split ({role: train|val|test, val_fraction: float,
                              test_fraction: float, seed: int, stratify: bool}).

    Returns:
        dict: loader name -> DataLoader
    """
    loaders = {}
    for loader_name, loader_cfg in datasets_cfg.items():
        dataset = DatasetBuilder(loader_cfg).build(DATA_DIR)
        data = dataset.load_source(loader_cfg.source)

        if split_cfg := loader_cfg.get('split'):
            data = _select_split(data, split_cfg)

        if loader_cfg.get('transforms'):
            data = _TransformedDataset(data, TransformBuilder(loader_cfg.transforms).build())

        loaders[loader_name] = DataLoader(
            data,
            batch_size=loader_cfg.batch_size,
            shuffle=loader_cfg.get('shuffle', False),
            num_workers=loader_cfg.num_workers,
            pin_memory=True,
        )
    return loaders


def _select_split(data, split_cfg):
    """Split a source dataset and return the subset for the configured role."""
    val_fraction = split_cfg.val_fraction
    test_fraction = split_cfg.get('test_fraction', 0.0)
    roles = ('train', 'val', 'test') if test_fraction else ('train', 'val')
    if split_cfg.role not in roles:
        raise ValueError(f"split.role must be one of {list(roles)}, got '{split_cfg.role}'")

    generator = torch.Generator().manual_seed(split_cfg.seed)
    if split_cfg.get('stratify', False):
        subsets = _stratified_subsets(data, val_fraction, test_fraction, generator)
    else:
        n_val, n_test = int(len(data) * val_fraction), int(len(data) * test_fraction)
        lengths = [len(data) - n_val - n_test, n_val] + ([n_test] if n_test else [])
        subsets = random_split(data, lengths, generator=generator)
    return dict(zip(roles, subsets))[split_cfg.role]


def _stratified_subsets(data, val_fraction, test_fraction, generator):
    """Shuffle each class's indices via `data.targets`, then split train/val(/test)."""
    targets = torch.as_tensor(data.targets)
    splits = ([], [], [])
    for class_label in targets.unique(sorted=True):
        indices = torch.nonzero(targets == class_label).squeeze(1)
        indices = indices[torch.randperm(len(indices), generator=generator)]
        n_val = int(len(indices) * val_fraction)
        n_test = int(len(indices) * test_fraction)
        n_train = len(indices) - n_val - n_test
        for split, part in zip(splits, indices.split([n_train, n_val, n_test])):
            split.append(part)
    return [Subset(data, torch.cat(parts).tolist()) for parts in splits]


class _TransformedDataset:
    """Applies a transform to a dataset subset."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return self.transform(x), y

    def __len__(self):
        return len(self.dataset)
