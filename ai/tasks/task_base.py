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
    # def process_vision(self, sub: 'Submarine', camera_image: 'pygame.Surface') -> Vision: 
    #     # This method is no longer called by execute. 
    #     # Tasks that need specific pre-processing before subtasks run can override execute.
    #     # However, the standard pattern now is for Submarine to call vision.update() once.
    #     return Vision(image_provider=lambda: camera_image) # Example default if needed, but likely unused
    # ---

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, 
                # --- vision_data argument type hint changed ---
                vision_data: Vision, 
                # ---
                config: SimulationConfig) -> Tuple[TaskStatus, ThrusterCommands]:
        """
        Manages subtask execution using shared context.
        Receives the Vision object (already updated) from Submarine.
        Commands spin in specified direction if WaitForTargetVisible is running.
        Propagates FAILED status from subtasks upwards.
        """
        if not self.subtasks: return TaskStatus.COMPLETED, ThrusterCommands()
        if self.current_subtask_index >= len(self.subtasks): return TaskStatus.COMPLETED, ThrusterCommands()

        # --- process_vision call is REMOVED ---
        # processed_vision_data = self.process_vision(sub, sensors.camera_image) 
        # --- The vision_data object passed in is already updated ---
        processed_vision_data = vision_data 
        # ---

        current_subtask = self.subtasks[self.current_subtask_index]

        if not hasattr(current_subtask, '_has_entered'):
             # Pass the already-updated vision object to on_enter
             current_subtask.on_enter(sub, sensors, processed_vision_data, self.context)
             current_subtask._has_entered = True

        # Pass the already-updated vision object to execute
        subtask_status, commands = current_subtask.execute(sub, dt, sensors, processed_vision_data, config, self.context)

        # Apply search spin if needed (logic remains the same, uses processed_vision_data)
        if isinstance(current_subtask, WaitForTargetVisible) and subtask_status == SubtaskStatus.RUNNING:
             # Check visibility using the passed Vision object's methods
             # Needs refinement: which target should WaitForTargetVisible wait for?
             # Assume pole OR gate for now. A better approach might involve context.
             if not (processed_vision_data.is_pole_visible() or processed_vision_data.is_gate_visible()):
                 spin_yaw = self.DEFAULT_SEARCH_TURN_POWER * -self.search_direction
                 commands = sub._mix_and_normalize_commands(0.0, 0.0, spin_yaw)
        
        # Subtask completion/failure logic remains the same
        if subtask_status == SubtaskStatus.COMPLETED:
            current_subtask.on_exit(sub, sensors, processed_vision_data, self.context)
            delattr(current_subtask, '_has_entered')
            self.current_subtask_index += 1
            if self.current_subtask_index >= len(self.subtasks): 
                # Task complete, return last command unless it was zero for transition
                final_commands = commands if (commands.port != 0 or commands.starboard != 0) else sub._get_damping_commands(sensors)
                return TaskStatus.COMPLETED, final_commands
            else:
                 # Start next subtask
                 next_subtask = self.subtasks[self.current_subtask_index]
                 next_subtask.on_enter(sub, sensors, processed_vision_data, self.context)
                 next_subtask._has_entered = True
                 # Return zero commands during subtask transition
                 return TaskStatus.RUNNING, ThrusterCommands() 

        elif subtask_status == SubtaskStatus.FAILED:
            print(f"ERROR: Subtask {current_subtask.name} FAILED in task {self.__class__.__name__}")
            current_subtask.on_exit(sub, sensors, processed_vision_data, self.context)
            return TaskStatus.FAILED, sub._get_damping_commands(sensors)

        # Subtask still running
        return TaskStatus.RUNNING, commands