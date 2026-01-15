# Make Ignite Engine State Persistent

## Problem

When resuming BNN training from a checkpoint, `engine.state.epoch` resets to 1, causing:

- Sample filename collisions (epoch 70 saves as `sample_0005.pt`)
- Overwriting old samples
- Incorrect W&B epoch logging

## Solution

Persist Ignite's `trainer.state.epoch` and `trainer.state.iteration` across training sessions by saving/restoring them in checkpoints.

## Implementation Steps

### 1. Save Iteration in Checkpoints

**File:** `src/training/handlers/checkpoint.py`**Best checkpoint (line 38-49):**

```python
checkpoint = {
    'epoch': trainer.state.epoch,
    'iteration': trainer.state.iteration,  # ADD THIS
    'model_state_dict': model.state_dict(),
    # ... rest unchanged
}
```

**Last checkpoint (line 73-82):**

```python
checkpoint = {
    'epoch': engine.state.epoch,
    'iteration': engine.state.iteration,  # ADD THIS
    'model_state_dict': model.state_dict(),
    # ... rest unchanged
}
```



### 2. Return Iteration from CheckpointManager

**File:** `src/utils/checkpoint_manager.py`**Update `load_checkpoint` method:**Change return signature (line 30):

```python
Returns:
    tuple: (start_epoch, start_iteration, sample_files)
```

Update no-checkpoint return (line 40):

```python
return 0, 0, None
```

Update checkpoint return (line 57-61):

```python
self.start_epoch = checkpoint['epoch']
self.wandb_run_id = checkpoint.get('wandb_run_id', None)
sample_files = checkpoint.get('sample_files', None)
start_iteration = checkpoint.get('iteration', 0)

return self.start_epoch, start_iteration, sample_files
```



### 3. Restore State in train.py

**File:** `train.py`**Update checkpoint loading (line 52):**

```python
start_epoch, start_iteration, sample_files = checkpoint_manager.load_checkpoint(model, optimizer, device, scheduler)
```

**Add state restoration after trainer build (after line 69):**

```python
trainer = TrainerBuilder(cfg.training).build(...)

# Restore trainer state for resumption
if start_epoch > 0:
    trainer.state.epoch = start_epoch
    trainer.state.iteration = start_iteration

# Restore sample files if resuming BNN training
if sample_files and hasattr(trainer, 'sample_collector') and trainer.sample_collector is not None:
    trainer.sample_collector.sample_files = sample_files
```

**Fix max_epochs calculation (line 76-78):**

```python
# OLD:
remaining_epochs = cfg.training.num_epochs - start_epoch
if remaining_epochs > 0:
    trainer.run(loaders['train'], max_epochs=remaining_epochs)

# NEW:
if start_epoch < cfg.training.num_epochs:
    trainer.run(loaders['train'], max_epochs=cfg.training.num_epochs)
```



## How It Works

**Initial training:**

- `trainer.state.epoch` starts at 0
- Runs epochs 1-100
- Saves: `sample_0050.pt`, `sample_0055.pt`, `sample_0060.pt`

**Resume at epoch 65:**

- Load checkpoint: `start_epoch=65, start_iteration=6500`
- Restore state: `trainer.state.epoch = 65`
- Run `trainer.run(max_epochs=100)` → Ignite runs epochs 66-100
- Saves: `sample_0070.pt`, `sample_0075.pt` ✓ (not `sample_0005.pt`!)

## Files Modified

1. `src/training/handlers/checkpoint.py` - Save iteration
2. `src/utils/checkpoint_manager.py` - Return iteration
3. `train.py` - Restore state and fix max_epochs

## Benefits

- No sample filename collisions
- Correct W&B epoch logging