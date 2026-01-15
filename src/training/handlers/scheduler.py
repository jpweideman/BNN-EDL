"""Learning rate scheduler handler."""

from ignite.engine import Events


def attach_scheduler_handler(trainer, scheduler):
    """
    Attach scheduler to update learning rate after each epoch.
    
    Args:
        trainer: Training engine
        scheduler: PyTorch scheduler instance
    """
    @trainer.on(Events.EPOCH_COMPLETED)
    def step_scheduler(engine):
        scheduler.step()

