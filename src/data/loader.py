"""DataLoader creation with transforms and splits."""

import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from src.builders.dataset_builder import DatasetBuilder
from src.builders.transform_builder import TransformBuilder


def create_loaders(dataset_config, seed):
    """
    Create data loaders for all configured splits.

    Args:
        dataset_config: Dataset configuration (splits, transforms, batch_size, etc.)
        seed: Random seed for reproducible data splitting

    Returns:
        dict: split name -> DataLoader
    """
    data_dir = str(Path(__file__).parent.parent.parent / "data")

    transforms_dict = {}
    for split_name, split_config in dataset_config.splits.items():
        if hasattr(split_config, 'transforms') and split_config.transforms:
            transforms_dict[split_name] = TransformBuilder(split_config.transforms).build()

    dataset = DatasetBuilder(dataset_config).build(data_dir)

    splits_by_source = {}
    for split_name, split_config in dataset_config.splits.items():
        splits_by_source.setdefault(split_config.source, []).append((split_name, split_config))

    for source, splits in splits_by_source.items():
        total = sum(s.fraction for _, s in splits)
        if total != 1.0:
            raise ValueError(f"Fractions for source '{source}' sum to {total:.4f}, must equal 1.0")

    loaders = {}
    for source, splits in splits_by_source.items():
        raw_data = dataset.load_source(source)
        fractions = [s.fraction for _, s in splits]
        subsets = random_split(raw_data, fractions, generator=torch.Generator().manual_seed(seed))

        for (split_name, split_config), subset in zip(splits, subsets):
            transform = transforms_dict.get(split_name)
            if transform:
                subset = _TransformedDataset(subset, transform)
            loaders[split_name] = DataLoader(
                subset,
                batch_size=dataset_config.batch_size,
                shuffle=split_config.get('shuffle', False),
                num_workers=dataset_config.num_workers,
                pin_memory=True
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
