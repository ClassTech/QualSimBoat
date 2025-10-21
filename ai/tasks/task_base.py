#!/usr/bin/env python3
"""
Base class definition for all autonomous mission tasks.
Includes subtask execution logic with a shared context.
Handles subtask failure by restarting the task sequence.
Allows tasks to specify search direction on restart.
"""
from enum import Enum, auto
from typing import Tuple, List, Dict, Any

from data_structures import SensorSuite, VisionData, ThrusterCommands
from config import SimulationConfig
from ai.tasks.subtask_base import Subtask, SubtaskStatus
from ai.tasks.common_subtasks import WaitForTargetVisible

# Forward declaration
class Submarine: pass

class TaskStatus(Enum):
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto() # Propagated up if task cannot recover

class Task:
    """Base class for a mission task with subtask execution."""
    subtasks: List[Subtask] = []
    current_subtask_index: int = 0
    DEFAULT_SEARCH_TURN_POWER = 0.25
    context: Dict[str, Any] = {}
    # --- NEW: Search direction (+1 Left, -1 Right) ---
    search_direction: int = 1 # Default to left spin
    # ---

    def reset(self, search_direction: int = 1): # Allow overriding search direction on reset
        """Resets the task and subtask states, clears context, sets search direction."""
        self.current_subtask_index = 0
        self.context = {}
        self.search_direction = search_direction # Set search direction
        for subtask in self.subtasks:
             if hasattr(subtask, '_has_entered'): delattr(subtask, '_has_entered')
             if hasattr(subtask, 'reset'): subtask.reset()

    @property
    def state_name(self) -> str: # ... (remains the same) ...
        if self.subtasks and 0 <= self.current_subtask_index < len(self.subtasks):
            subtask_name = self.subtasks[self.current_subtask_index].name
            if hasattr(self.subtasks[self.current_subtask_index], 'get_dynamic_name'):
                subtask_name = self.subtasks[self.current_subtask_index].get_dynamic_name(self.context)
            return f"{self.__class__.__name__}[{self.current_subtask_index}]:{subtask_name}"
        return self.__class__.__name__

    def process_vision(self, sub: 'Submarine', camera_image: 'pygame.Surface') -> VisionData: # ... (remains the same) ...
        return VisionData()

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data_ignored: VisionData, config: SimulationConfig) -> Tuple[TaskStatus, ThrusterCommands]:
        """
        Runs process_vision, manages subtask execution using shared context.
        Commands spin in specified direction if WaitForTargetVisible is running.
        Propagates FAILED status from subtasks upwards.
        """
        if not self.subtasks: return TaskStatus.COMPLETED, ThrusterCommands()
        if self.current_subtask_index >= len(self.subtasks): return TaskStatus.COMPLETED, ThrusterCommands()

        processed_vision_data = self.process_vision(sub, sensors.camera_image)
        current_subtask = self.subtasks[self.current_subtask_index]

        if not hasattr(current_subtask, '_has_entered'):
             current_subtask.on_enter(sub, sensors, processed_vision_data, self.context)
             current_subtask._has_entered = True

        subtask_status, commands = current_subtask.execute(sub, dt, sensors, processed_vision_data, config, self.context)

        # --- MODIFIED: Use self.search_direction for spin ---
        if isinstance(current_subtask, WaitForTargetVisible) and subtask_status == SubtaskStatus.RUNNING:
             if not processed_vision_data.gate_is_visible:
                 # Apply turn power in the specified direction
                 spin_yaw = self.DEFAULT_SEARCH_TURN_POWER * self.search_direction
                 commands = sub._mix_and_normalize_commands(0.0, 0.0, spin_yaw)
        # --- END MODIFICATION ---

        if subtask_status == SubtaskStatus.COMPLETED:
             # (Completion logic remains the same)
            current_subtask.on_exit(sub, sensors, processed_vision_data, self.context)
            delattr(current_subtask, '_has_entered')
            self.current_subtask_index += 1
            if self.current_subtask_index >= len(self.subtasks): return TaskStatus.COMPLETED, commands
            else:
                 next_subtask = self.subtasks[self.current_subtask_index]
                 next_subtask.on_enter(sub, sensors, processed_vision_data, self.context)
                 next_subtask._has_entered = True
                 return TaskStatus.RUNNING, ThrusterCommands()

        elif subtask_status == SubtaskStatus.FAILED:
            # (Failure logic remains the same - propagate up)
        #    print(f"ERROR: Subtask {current_subtask.name} FAILED in task {self.__class__.__name__}")
            current_subtask.on_exit(sub, sensors, processed_vision_data, self.context)
            return TaskStatus.FAILED, sub._get_damping_commands(sensors)

        return TaskStatus.RUNNING, commands