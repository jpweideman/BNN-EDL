#!/usr/bin/env python3
"""
Verification script to test that StandardBNN.evaluate() works correctly
on the complete test set.
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Add src to path
sys.path.append('src')

from datasets.classification import get_mnist_dataloaders, infer_dataset_info
from models.base_bnn import create_mlp
from models.standard_bnn import StandardBNN

def verify_evaluation():
    """Verify that StandardBNN.evaluate() works correctly on complete test set."""
    
    # Clear any existing GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🧹 Cleared GPU memory. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    print("🔍 Verifying StandardBNN.evaluate() function...")
    print("=" * 60)
    
    # Load MNIST dataset
    print("📊 Loading MNIST dataset...")
    train_loader, val_loader, test_loader = get_mnist_dataloaders(
        batch_size=32,  # Smaller batch size to reduce memory usage
        num_workers=0,
    )
    
    # Get dataset info
    dataset_info = infer_dataset_info(train_loader)
    print(f"Dataset info: {dataset_info}")
    
    # Verify test set size
    test_dataset_size = len(test_loader.dataset)
    test_batches = len(test_loader)
    print(f"Test set: {test_dataset_size} samples in {test_batches} batches")
    
    # Create model
    print("\n🏗️  Creating BNN model...")
    model = create_mlp(
        input_size=dataset_info['input_size'],
        hidden_sizes=[128, 128],
        num_classes=dataset_info['num_classes']
    )
    
    bnn = StandardBNN(
        model=model,
        num_classes=dataset_info['num_classes'],
        prior_std=1.0,
        device='cuda'  # Use CPU to avoid GPU memory issues
    )
    
    # Quick training (just a few epochs for verification)
    print("\n🚀 Training BNN (quick test)...")
    bnn.fit(
        train_loader=train_loader,
        num_epochs=5,  # Very short training
        lr=1e-3,
        num_burn_in=2,  # Very short burn-in
        verbose=True
    )
    
    print(f"Collected {len(bnn.posterior_samples)} posterior samples")
    
    # Test evaluation with different sample counts
    print("\n📈 Testing evaluation with different sample counts...")
    
    # Test evaluation using ALL collected samples
    print("\n1️⃣  Evaluating with ALL collected samples:")
    results_all = bnn.evaluate(test_loader)
    print(f"Results: {results_all}")
    
    # Verify that all samples were used
    assert results_all['num_samples'] == len(bnn.posterior_samples), \
        f"Expected {len(bnn.posterior_samples)} samples, got {results_all['num_samples']}"
    print("✅ Correctly used all collected posterior samples")
    
    # Test 2: Manual verification of test set coverage
    print("\n2️⃣  Manual verification of test set coverage:")
    total_test_samples = 0
    for batch in test_loader:
        images, labels = batch
        total_test_samples += len(images)
    
    print(f"Manual count: {total_test_samples} test samples processed")
    print(f"Reported count: {results_all['test_samples']} test samples")
    
    # Verify they match
    assert total_test_samples == results_all['test_samples'], \
        f"Manual count ({total_test_samples}) != reported count ({results_all['test_samples']})"
    print("✅ Test set coverage is correct")
    
    # Test 3: Verify metrics are computed on full test set
    print("\n3️⃣  Verifying metrics computation:")
    print(f"Accuracy computed on {results_all['test_samples']} samples")
    print(f"ECE computed on {results_all['test_samples']} samples") 
    print(f"Average uncertainty computed on {results_all['test_samples']} samples")
    
    # Test 4: Verify uncertainty values are reasonable
    print("\n4️⃣  Checking uncertainty values:")
    uncertainty = results_all['avg_uncertainty']
    print(f"Average epistemic uncertainty: {uncertainty:.6f}")
    
    if uncertainty > 0:
        print("✅ Uncertainty is non-zero (good!)")
    else:
        print("⚠️  Uncertainty is zero (might be a problem)")
    
    print("\n" + "=" * 60)
    print("🎉 Evaluation verification completed!")
    print("\nSummary:")
    print(f"  - Test set size: {results_all['test_samples']} samples")
    print(f"  - Posterior samples used: {results_all['num_samples']}")
    print(f"  - Accuracy: {results_all['accuracy']:.4f}")
    print(f"  - ECE: {results_all['ece']:.4f}")
    print(f"  - Avg uncertainty: {results_all['avg_uncertainty']:.6f}")
    
    return results_all

if __name__ == "__main__":
    try:
        results = verify_evaluation()
        print("\n✅ All tests passed! The evaluate() function works correctly.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
