"""Weight loader for loading pretrained model weights."""

import torch
from pathlib import Path


class WeightLoaderSetup:
    """Orchestrates loading of pretrained model weights."""
    
    def load_pretrained_weights(self, model, pretrained_path, device):
        """
        Load pretrained model weights (e.g., from standard training).
        
        Args:
            model: Model to load weights into
            pretrained_path: Path to pretrained checkpoint or model weights
            device: Device to map to
        
        Raises:
            FileNotFoundError: If pretrained_path does not exist
        """
        pretrained_path = Path(pretrained_path)
        
        if not pretrained_path.exists():
            raise FileNotFoundError(f"Pretrained model not found at {pretrained_path}")
        
        print(f"Loading pretrained model from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        
        # Handle both raw state_dict and checkpoint format
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
