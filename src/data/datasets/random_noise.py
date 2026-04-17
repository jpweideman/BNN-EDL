"""Random noise dataset for OOD evaluation."""

import torch
from src.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("random_noise")
class RandomNoiseDataset:
    """Generates uniform random noise tensors as a synthetic OOD dataset.

    Args:
        data_dir: Ignored, present for interface consistency.
        shape: Tensor shape per sample, e.g. [1, 28, 28].
        size: Number of samples.
        num_classes: Number of target classes (labels are assigned uniformly at random).
    """

    def __init__(self, data_dir, shape, size, num_classes, seed=0):
        self.shape = shape
        self.size = size
        self.num_classes = num_classes
        self.seed = seed

    def load_source(self, source_name):
        return _RandomNoiseSource(self.shape, self.size, self.num_classes, self.seed)


class _RandomNoiseSource:
    def __init__(self, shape, size, num_classes, seed):
        rng = torch.Generator().manual_seed(seed)
        self.data = torch.rand(size, *shape, generator=rng)
        self.targets = torch.randint(0, num_classes, (size,), generator=rng)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx].item()

    def __len__(self):
        return len(self.data)
