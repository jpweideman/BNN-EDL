"""Progress bar handler."""

from ignite.contrib.handlers import ProgressBar


def attach_progress_bar_to_engine(engine, show_loss=True):
    """
    Attach progress bar to any engine.
    
    Args:
        engine: Ignite engine
        show_loss: Whether to show loss/metrics in progress bar
    """
    pbar = ProgressBar()
    if show_loss:
        pbar.attach(engine, output_transform=lambda output: output)
    else:
        pbar.attach(engine)

