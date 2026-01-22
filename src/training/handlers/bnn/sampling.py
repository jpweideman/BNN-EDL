"""Sampling handler for BNN."""

import torch
from pathlib import Path
from ignite.engine import Events

class SamplingManager:
    """Manages parameter samples during SGMCMC training."""
    
    def __init__(self, output_dir, config):
        self.samples_dir = Path(output_dir) / "samples"
        self.samples_dir.mkdir(exist_ok=True, parents=True)
        self.config = config
        self.sample_files = []
        
    def attach(self, trainer, model):
        """Attach to trainer."""
        @trainer.on(Events.EPOCH_COMPLETED)
        def save_sample(engine):
            epoch = engine.state.epoch
            
            if epoch >= self.config.start_epoch and \
               (epoch - self.config.start_epoch) % self.config.sample_interval == 0:
                
                sample_path = self.samples_dir / f"sample_{epoch:04d}.pt"
                torch.save(model.state_dict(), sample_path)
                self.sample_files.append(sample_path)
    
    def get_sample_files(self):
        """Return list of collected sample file paths."""
        return self.sample_files

