#!/usr/bin/env python3
"""
A simple task to bring the submarine to a full stop using PID control.
Now uses the Vision class.
"""
import math
from typing import Tuple

from .task_base import Task, TaskStatus
# --- CORRECT IMPORT ---
from data_structures import SensorSuite, Vision, ThrusterCommands
# ---
from config import SimulationConfig

class StabilizeTask(Task):
    """Holds position and heading using PID control."""
    def __init__(self, duration: float = 3.0, speed_threshold: float = 0.05):
        self.STABILIZE_DURATION = duration
        self.SPEED_THRESHOLD = speed_threshold
        # Internal state
        self.state_timer = 0.0
        self.target_set = False
        self._current_speed = 0.0 # Store for state_name
        super().__init__() # Call base __init__
        self.reset() # Call our reset

    def reset(self, search_direction: int = 1): # Match base signature
        super().reset(search_direction) # Call base reset
        self.state_timer = 0.0
        self.target_set = False
        self._current_speed = 0.0

    @property
    def state_name(self) -> str:
        # Access _current_speed safely
        speed = getattr(self, '_current_speed', 0.0)
        return f"STABILIZING (Spd: {speed:.2f} m/s)"

    # process_vision is removed as it's handled by the Vision class

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, 
                vision_data: Vision, # Correct type hint
                config: SimulationConfig) -> Tuple[TaskStatus, ThrusterCommands]:
        
        # On first execution, lock position/heading and RESET PID INTEGRALS.
        if not self.target_set:
            sub.target_x, sub.target_y = sensors.x, sensors.y
            sub.target_heading = sensors.heading
            # sub.target_pitch = 0.0 # Pitch not controlled here
            sub.integral_x_err, sub.integral_y_err = 0.0, 0.0
            self.target_set = True

        self.state_timer += dt
        speed = math.hypot(sensors.velocity_x, sensors.velocity_y)
        self._current_speed = speed # Update for state_name

        # Check completion
        if self.state_timer > self.STABILIZE_DURATION and speed < self.SPEED_THRESHOLD:
            return TaskStatus.COMPLETED, sub._get_damping_commands(sensors)

        # Execute PID hover control
        return TaskStatus.RUNNING, sub._get_pid_hover_commands(sensors, dt, sub.target_x, sub.target_y)