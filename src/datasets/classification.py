import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple, Dict, Any
import os


def get_mnist_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 32,
    train_split: float = 0.8,
    normalize: bool = True,
    flatten: bool = True,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get MNIST dataloaders for train, validation, and test.
    
    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size for dataloaders
        train_split: Fraction of training data to use for training, rest for validation
        normalize: Whether to normalize pixel values
        flatten: Whether to flatten images to vectors
        num_workers: Number of parallel CPU workers for data loading. 
            Pre-loads batches while the model trains on GPU/CPU.  
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    
    Note:
        The test set is the official MNIST test split of 10k samples.
        train_split only affects the train/val split of the 60k training set.
    """
    # Define transforms
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))
    
    transform = transforms.Compose(transform_list)
    
    # Download/load datasets
    train_dataset = datasets.MNIST(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Split training data into train and validation
    train_size = int(train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(1)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def get_cifar10_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 32,
    train_split: float = 0.8,
    normalize: bool = True,
    num_workers: int = 2,
    augment: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get CIFAR-10 dataloaders for train, validation, and test.
    
    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size for dataloaders
        train_split: Fraction of training data to use for training, rest for validation
        normalize: Whether to normalize pixel values
        num_workers: Number of parallel CPU workers for data loading. 
            Pre-loads batches while the model trains on GPU/CPU.
        augment: Whether to apply data augmentation to training data
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    
    Note:
        The test set is the official CIFAR-10 test split of 10k samples.
        train_split only affects the train/val split of the 50k training set.
    """
    # Define training transforms (with augmentation)
    train_transform_list = []
    if augment:
        train_transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4)
            
        ])
    train_transform_list.append(transforms.ToTensor())
    
    if normalize:
        # Use normalization to match https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/trainer.py
        train_transform_list.append(
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
            )
        )
    
    train_transform = transforms.Compose(train_transform_list)
    
    # Define test transforms (no augmentation)
    test_transform_list = [transforms.ToTensor()]
    if normalize:
        test_transform_list.append(
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
            )
        )
    
    test_transform = transforms.Compose(test_transform_list)
    
    # Download/load datasets
    train_dataset = datasets.CIFAR10(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=test_transform
    )
    
    # Split training data into train and validation
    train_size = int(train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(1)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def infer_dataset_info(dataloader: DataLoader) -> Dict[str, Any]:
    """
    Get input shape, flattened input size, and number of classes from a dataloader.
    
    Args:
        dataloader: PyTorch DataLoader to inspect
    
    Returns:
        Dictionary with dataset information:
            - input_shape: tuple of sample dimensions 
            - input_size: int of flattened feature dimension
            - num_classes: int of unique classes in the dataset
    """
    # Get the dataset and create a full scan loader
    dataset = dataloader.dataset
    scan_loader = DataLoader(
        dataset,
        batch_size=dataloader.batch_size or 1024,
        shuffle=False,
        num_workers=getattr(dataloader, 'num_workers', 0)
    )
    
    # Input shape from first batch
    first_batch = next(iter(scan_loader))
    images, _ = first_batch
    input_shape = tuple(images.shape[1:])  # Exclude batch dimension
    input_size = int(torch.tensor(input_shape).prod().item())  # Flattened size
    
    # Collect unique labels from all batches
    uniques = set()
    for _, labels in scan_loader:
        uniques.update(torch.unique(labels).tolist())
    
    num_classes = len(uniques)

    return {
        'input_shape': input_shape,
        'input_size': input_size,
        'num_classes': num_classes,
    }


 
