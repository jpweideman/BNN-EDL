"""Device setup utilities."""

import torch


def setup_device(device_config):
    """
    Setup device based on config.
    
    Args:
        device_config: 'auto', 'cuda', 'cpu', or 'mps'
    
    Returns:
        torch.device
    """
    if device_config == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_config)
    
    print(f"Using device: {device}")
    
    return device

