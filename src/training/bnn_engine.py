"""BNN-specific training and evaluation engines."""

import torch
from ignite.engine import Engine

def create_bnn_train_engine(model, optimizer, device):
    """
    Create BNN training engine.
    
    Args:
        model: PyTorch model
        optimizer: BNN optimizer 
        device: Device to run on
    
    Returns:
        Ignite Engine for BNN training
    """
    def bnn_train_step(engine, batch):
        model.train()
        
        x, y = batch
        x, y = x.to(device), y.to(device)
        
        # Posteriors sampler step
        optimizer.step((x, y))
        
        # Get BNN metrics for logging
        metrics = optimizer.get_last_metrics()
        return metrics 
    
    return Engine(bnn_train_step)


def create_bnn_eval_engine(model, criterion, device):
    """
    Create BNN evaluation engine.
    
    Args:
        model: PyTorch model
        criterion: Loss function
        device: Device to run on
    
    Returns:
        Ignite Engine for BNN evaluation
        
    Note:
        The sampling_manager will be set on the engine by the trainer after creation.
    """
    
    def bnn_eval_step(engine, batch):
        model.eval()
        x, y = batch
        x, y = x.to(device), y.to(device)
        
        with torch.no_grad():
            # Get sampling_manager from engine (set by trainer)
            sampling_manager = getattr(engine, 'sampling_manager', None)
            current_sample_files = sampling_manager.get_sample_files() if sampling_manager else []
            
            # Check if samples are available
            if current_sample_files and len(current_sample_files) > 0:
                # Ensemble evaluation - collect predictions from all samples
                all_preds = []
                for sample_file in current_sample_files:
                    state_dict = torch.load(sample_file, map_location=device)
                    model.load_state_dict(state_dict)
                    pred = model(x)
                    all_preds.append(pred)
                
                all_preds = torch.stack(all_preds)
                
                # Use current model state for standard metrics
                y_pred = model(x)
                loss = criterion(y_pred, y)
                
                # Return all_preds for BNN ensemble metrics to compute BMA
                return {
                    'y_pred': y_pred,
                    'y': y,
                    'loss': loss,
                    'all_preds': all_preds
                }
            else:
                # Single model evaluation (no samples yet)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                return {'y_pred': y_pred, 'y': y, 'loss': loss}
    
    engine = Engine(bnn_eval_step)
    return engine

