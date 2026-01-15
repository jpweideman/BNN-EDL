"""Ignite training and evaluation engines."""

import torch
from ignite.engine import Engine


def create_train_engine(model, optimizer, criterion, device):
    """
    Create training engine.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: Device to run on
    
    Returns:
        Ignite Engine for training
    """
    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        
        x, y = batch
        x, y = x.to(device), y.to(device)
        
        y_pred = model(x)
        loss = criterion(y_pred, y)
        
        loss.backward()
        optimizer.step()
        
        return {'loss': loss.item()}
    
    return Engine(train_step)


def create_eval_engine(model, criterion, device):
    """
    Create evaluation engine.
    
    Args:
        model: PyTorch model
        criterion: Loss function
        device: Device to run on
    
    Returns:
        Ignite Engine for evaluation
    """
    def eval_step(engine, batch):
        model.eval()
        
        with torch.no_grad():
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            y_pred = model(x)
            loss = criterion(y_pred, y)
            
            return {'y_pred': y_pred, 'y': y, 'loss': loss}
    
    return Engine(eval_step)

