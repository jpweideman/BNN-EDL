"""Utilities for loading pretrained model weights."""

import torch
from pathlib import Path


def load_pretrained_weights(model, pretrained_path, device):
    """
    Load pretrained model weights into model.

    Args:
        model: Model to load weights into
        pretrained_path: Path to pretrained checkpoint or state dict
        device: Device to map to

    Raises:
        FileNotFoundError: If pretrained_path does not exist
    """
    pretrained_path = Path(pretrained_path)
    if not pretrained_path.exists():
        raise FileNotFoundError(f"Pretrained model not found at {pretrained_path}")

    print(f"Loading pretrained model from {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
