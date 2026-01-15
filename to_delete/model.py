"""Model setup for creating and initializing models."""

import torch
from pathlib import Path
from src.builders.model_builder import ModelBuilder


class ModelSetup:
    """Orchestrates model creation and initialization."""
    
    def create_model(self, model_config, device):
        """
        Create and move model to device.
        
        Args:
            model_config: Model configuration from Hydra
            device: Device to move model to
        
        Returns:
            Model on specified device
        """
        model = ModelBuilder(model_config).build()
        return model.to(device)
    
    def load_pretrained_weights(self, model, pretrained_path, device):
        """
        Load pretrained model weights (e.g., from standard training).
        
        Args:
            model: Model to load weights into
            pretrained_path: Path to pretrained checkpoint or model weights
            device: Device to map to
        
        Returns:
            bool: True if loaded successfully
        """
        pretrained_path = Path(pretrained_path)
        
        if not pretrained_path.exists():
            print(f"Pretrained model not found at {pretrained_path}")
            return False
        
        print(f"Loading pretrained model from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        
        # Handle both raw state_dict and checkpoint format
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print("Pretrained model loaded successfully. Starting fresh training.")
        return True

