"""Base builder class for all builders."""

from abc import ABC, abstractmethod
from typing import Any

class BaseBuilder(ABC):
    """
    Abstract base class for all builders.
    
    Builders construct objects from configuration in a consistent way.
    """
    
    def __init__(self, config: Any):
        """
        Initialize builder with configuration.
        
        Args:
            config: Configuration object (can be dict, DictConfig, or dataclass)
        """
        self.config = config
    
    @abstractmethod
    def build(self):
        """
        Build the object from configuration.
        
        Returns:
            The constructed object
        """
        pass

