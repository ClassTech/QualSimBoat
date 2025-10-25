#!/usr/bin/env python3
"""
Base class definition for all autonomous mission tasks.
Includes subtask execution logic with a shared context.
Handles subtask failure by restarting the task sequence.
Allows tasks to specify search direction on restart.
"""
from enum import Enum, auto
from typing import Tuple, List, Dict, Any

# --- Import Vision class instead of VisionData ---
from data_structures import SensorSuite, Vision, ThrusterCommands 
# ---
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
    search_direction: int = 1 

    def reset(self, search_direction: int = 1): 
        self.current_subtask_index = 0
        self.context = {}
        self.search_direction = search_direction 
        for subtask in self.subtasks:
             if hasattr(subtask, '_has_entered'): delattr(subtask, '_has_entered')
             if hasattr(subtask, 'reset'): subtask.reset()

    @property
    def state_name(self) -> str: 
        if self.subtasks and 0 <= self.current_subtask_index < len(self.subtasks):
            subtask_name = self.subtasks[self.current_subtask_index].name
            if hasattr(self.subtasks[self.current_subtask_index], 'get_dynamic_name'):
                subtask_name = self.subtasks[self.current_subtask_index].get_dynamic_name(self.context)
            return f"{self.__class__.__name__}[{self.current_subtask_index}]:{subtask_name}"
        return self.__class__.__name__

    # --- process_vision is REMOVED ---

    # --- MODIFIED: Signature updated ---
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, 
                front_vision: Vision, side_vision: Vision,
                config: SimulationConfig) -> Tuple[TaskStatus, ThrusterCommands]:
        """
        Manages subtask execution using shared context.
        Receives the Vision objects (already updated) from Submarine.
        Commands spin in specified direction if WaitForTargetVisible is running.
        Propagates FAILED status from subtasks upwards.
        """
        if not self.subtasks: return TaskStatus.COMPLETED, ThrusterCommands()
        if self.current_subtask_index >= len(self.subtasks): return TaskStatus.COMPLETED, ThrusterCommands()

        # --- process_vision call is REMOVED ---
        
        current_subtask = self.subtasks[self.current_subtask_index]

        if not hasattr(current_subtask, '_has_entered'):
             # --- MODIFIED: Pass both vision objects ---
             current_subtask.on_enter(sub, sensors, front_vision, side_vision, self.context)
             current_subtask._has_entered = True

        # --- MODIFIED: Pass both vision objects ---
        subtask_status, commands = current_subtask.execute(sub, dt, sensors, front_vision, side_vision, config, self.context)

        # Apply search spin if needed (logic remains the same, uses front_vision)
        if isinstance(current_subtask, WaitForTargetVisible) and subtask_status == SubtaskStatus.RUNNING:
             # Check visibility using the passed Vision object's methods
             # We assume WaitForTargetVisible always uses the FRONT camera
             if not (front_vision.is_pole_visible() or front_vision.is_gate_visible()):
                 spin_yaw = self.DEFAULT_SEARCH_TURN_POWER * -self.search_direction
                 commands = sub._mix_and_normalize_commands(0.0, 0.0, spin_yaw)
        
        # Subtask completion/failure logic
        if subtask_status == SubtaskStatus.COMPLETED:
            # --- MODIFIED: Pass both vision objects ---
            current_subtask.on_exit(sub, sensors, front_vision, side_vision, self.context)
            delattr(current_subtask, '_has_entered')
            self.current_subtask_index += 1
            if self.current_subtask_index >= len(self.subtasks): 
                # Task complete, return last command unless it was zero for transition
                final_commands = commands if (commands.port != 0 or commands.starboard != 0) else sub._get_damping_commands(sensors)
                return TaskStatus.COMPLETED, final_commands
            else:
                 # Start next subtask
                 next_task = self.subtasks[self.current_subtask_index]
                 # --- MODIFIED: Pass both vision objects ---
                 next_task.on_enter(sub, sensors, front_vision, side_vision, self.context)
                 next_task._has_entered = True
                 # Return zero commands during subtask transition
                 return TaskStatus.RUNNING, ThrusterCommands() 

        elif subtask_status == SubtaskStatus.FAILED:
            print(f"ERROR: Subtask {current_subtask.name} FAILED in task {self.__class__.__name__}")
            # --- MODIFIED: Pass both vision objects ---
            current_subtask.on_exit(sub, sensors, front_vision, side_vision, self.context)
            return TaskStatus.FAILED, sub._get_damping_commands(sensors)

        # Subtask still running
        return TaskStatus.RUNNING, commands