#!/usr/bin/env python3
"""
Implements a "Ratchet Turn" task using subtasks.
Aligns initially, surges past, yaws back, aligns subsequently, loops.
Completes when the gate is seen. Loop logic handled within the task.
"""
import math
from typing import Tuple, Dict, Any
import pygame
import numpy as np

# Absolute imports
from ai.tasks.task_base import Task, TaskStatus
from ai.tasks.subtask_base import SubtaskStatus 
from ai.tasks.common_subtasks import (WaitForTargetVisible, AlignToObjectX, 
                                      DriveUntilTargetLostRight, 
                                      SpinUntilTargetReappearsLeft)

# --- Import Vision class instead of VisionData ---
from data_structures import SensorSuite, Vision, ThrusterCommands 
# ---
from config import SimulationConfig # Keep config for subtask calls
# Vision processing functions no longer needed here
# from ai.vision import find_blobs_hsv 
# from config import GREEN_HSV_RANGE, RED_HSV_RANGES 
from utils import angle_diff

class RatchetTurnTask(Task):
    """
    Navigates around the pole using subtasks for align-surge-yaw-repeat
    until the red gate is detected. Loop logic handled in execute.
    """
    def __init__(self, 
                 initial_target_fraction=0.6, 
                 subsequent_target_fraction=0.8, 
                 surge_power=0.3, 
                 yaw_power=-0.3): 
        
        self.initial_target_fraction = initial_target_fraction
        self.subsequent_target_fraction = subsequent_target_fraction
        self.surge_power = surge_power
        self.yaw_power = yaw_power 
        
        align_tolerance_px = 15
        align_yaw_gain = 1.5 
        align_yaw_rate_tolerance = 0.05
        disappear_threshold_fraction = 0.95

        super().__init__() 

        self.loop_start_index = 2 
        self.subtasks = [
            WaitForTargetVisible(), 
            AlignToObjectX(target_x_fraction=self.initial_target_fraction, 
                           tolerance_px=align_tolerance_px, 
                           yaw_gain=align_yaw_gain, 
                           yaw_rate_tolerance=align_yaw_rate_tolerance), 
            DriveUntilTargetLostRight(surge_power=self.surge_power, 
                                      disappear_threshold_fraction=disappear_threshold_fraction), 
            SpinUntilTargetReappearsLeft(target_fraction=self.subsequent_target_fraction, 
                                         yaw_power=self.yaw_power), 
            AlignToObjectX(target_x_fraction=self.subsequent_target_fraction, 
                           tolerance_px=align_tolerance_px, 
                           yaw_gain=align_yaw_gain, 
                           yaw_rate_tolerance=align_yaw_rate_tolerance), 
        ]

        # No local vision data storage needed anymore
        # self.pole_vision_data = Vision() 
        # self.gate_vision_data = Vision()

        self.reset() 

    # --- process_vision method is REMOVED ---

    # Override execute to add the gate completion check AND loop logic
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, 
                # --- vision_data argument type hint changed ---
                vision_data: Vision, 
                # ---
                config: SimulationConfig) -> Tuple[TaskStatus, ThrusterCommands]:
        
        index_before_execute = self.current_subtask_index
        
        # Base class calls subtask execute, passing the updated vision_data object.
        status, commands = super().execute(sub, dt, sensors, vision_data, config) 
        
        # --- Check gate visibility using the passed vision_data object ---
        if index_before_execute > 0 and vision_data.is_gate_visible():
        # ---
            print("INFO: RatchetTurnTask - Gate Found! Completing task.")
            return TaskStatus.COMPLETED, sub._get_damping_commands(sensors)
        
        # --- Loop Logic (remains the same) ---
        if status == TaskStatus.COMPLETED and self.current_subtask_index >= len(self.subtasks):
            self.current_subtask_index = self.loop_start_index
            if self.subtasks and 0 <= self.current_subtask_index < len(self.subtasks):
                next_subtask = self.subtasks[self.current_subtask_index]
                # Re-create vision object? No, the Submarine object's instance is updated.
                # Just pass the existing, updated vision_data object to on_enter.
                next_subtask.on_enter(sub, sensors, vision_data, self.context)
                next_subtask._has_entered = True 
            return TaskStatus.RUNNING, ThrusterCommands() 
        # --- End Loop Logic ---
            
        return status, commands