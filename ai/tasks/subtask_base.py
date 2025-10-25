#!/usr/bin/env python3
"""
Base class for reusable sub-actions within a larger Task.
"""
from enum import Enum, auto
# Import TYPE_CHECKING
from typing import Tuple, Dict, Any, TYPE_CHECKING

# Import Vision etc.
from data_structures import SensorSuite, Vision, ThrusterCommands
from config import SimulationConfig

# Conditionally import Submarine ONLY for type checking
if TYPE_CHECKING:
    # Use absolute import path assuming 'ai' is a top-level package or visible from root
    from ai.submarine import Submarine

class SubtaskStatus(Enum):
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()

class Subtask:
    """Base class for a reusable action within a larger Task."""

    # Keep the forward reference string 'Submarine'
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite,
                 front_vision: Vision, side_vision: Vision,
                 context: Dict[str, Any]):
        pass

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite,
                front_vision: Vision, side_vision: Vision,
                config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        raise NotImplementedError

    def on_exit(self, sub: 'Submarine', sensors: SensorSuite,
                front_vision: Vision, side_vision: Vision,
                context: Dict[str, Any]):
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def get_dynamic_name(self, context: Dict[str, Any]) -> str:
        return self.name