"""DataLoader creation from named loader configurations."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

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
                      split ({role: train|val, val_fraction: float, seed: int}).

    Returns:
        dict: loader name -> DataLoader
    """
    loaders = {}
    for loader_name, loader_cfg in datasets_cfg.items():
        dataset = DatasetBuilder(loader_cfg).build(DATA_DIR)
        data = dataset.load_source(loader_cfg.source)

        if split_cfg := loader_cfg.get('split'):
            n_val = int(len(data) * split_cfg.val_fraction)
            train_sub, val_sub = random_split(
                data, [len(data) - n_val, n_val],
                generator=torch.Generator().manual_seed(split_cfg.seed),
            )
            if split_cfg.role == 'train':
                data = train_sub
            elif split_cfg.role == 'val':
                data = val_sub
            else:
                raise ValueError(f"split.role must be 'train' or 'val', got '{split_cfg.role}'")

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
