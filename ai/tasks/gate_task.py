#!/usr/bin/env python3
"""
Refactored Gate task using reusable subtasks. Includes stabilization after alignment.
Vision processing is now handled by the Vision class passed from Submarine.
"""
import math
from typing import Tuple, List
import pygame
import numpy as np

from ai.tasks.task_base import Task, TaskStatus
from ai.tasks.subtask_base import Subtask, SubtaskStatus
# --- Ensure DriveUntilTargetLost is imported ---
from ai.tasks.common_subtasks import (WaitForTargetVisible, AlignToObjectX, DriveStraight,
                                      Stabilize, DriveUntilTargetLost) 
# ---
from data_structures import SensorSuite, Vision, ThrusterCommands 
from utils import angle_diff


class GateTask(Task):
    """Navigates through the gate defined by two red poles."""

    def __init__(self, target_depth: float = 0.1):
        
        self.subtasks = [
            # Ensure WaitForTargetVisible checks for 'gate'
            WaitForTargetVisible(target_type='gate'), 
            AlignToObjectX(target_x_fraction=0.5,
                           tolerance_px=10,
                           yaw_gain=1.5, 
                           yaw_rate_tolerance=0.05), 
            Stabilize(duration=1.0, speed_threshold=0.1), 
            DriveUntilTargetLost(surge_power=0.6), # This import should now work
            DriveStraight(duration=4.0, surge_power=0.6), 
            Stabilize(duration=1.0) 
        ]
        
        super().__init__()
        self.reset()

    # --- process_vision method is REMOVED ---

    # Execute method is inherited