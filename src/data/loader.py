"""DataLoader creation from named loader configurations."""

from pathlib import Path
from torch.utils.data import DataLoader
from src.builders.dataset_builder import DatasetBuilder
from src.builders.transform_builder import TransformBuilder


def create_loaders(datasets_cfg):
    """
    Create one DataLoader per named loader config.

    Args:
        datasets_cfg: DictConfig mapping loader name to loader config.
            Each entry must have: name, source, batch_size, num_workers.
            Optional: shuffle (default False), transforms.

    Returns:
        dict: loader name -> DataLoader
    """
    data_dir = str(Path(__file__).parent.parent.parent / "data")
    loaders = {}
    for loader_name, loader_cfg in datasets_cfg.items():
        dataset = DatasetBuilder(loader_cfg).build(data_dir)
        data = dataset.load_source(loader_cfg.source)
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
