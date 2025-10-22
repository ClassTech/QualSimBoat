#!/usr/bin/env python3
"""
Refactored Gate task using reusable subtasks. Includes stabilization after alignment.
"""
import math
from typing import Tuple, List
import pygame
import numpy as np

from ai.tasks.task_base import Task, TaskStatus
from ai.tasks.subtask_base import Subtask, SubtaskStatus
# Import Stabilize from common_subtasks
from ai.tasks.common_subtasks import (WaitForTargetVisible, AlignToObjectX, DriveStraight,
                                      Stabilize, DriveUntilTargetLost)

from data_structures import SensorSuite, VisionData, ThrusterCommands
# --- MODIFICATION: Import RED_HSV_RANGES instead of GRAY_HSV_RANGE ---
from config import SimulationConfig, RED_HSV_RANGES # Was GRAY_HSV_RANGE
from utils import angle_diff
from ai.vision import find_blobs_hsv
# --- THE PROBLEMATIC IMPORT 'from ai.submarine import Submarine' IS REMOVED ---


class GateTask(Task):
    """Navigates through the gate defined by two red poles."""

    def __init__(self, target_depth: float = 0.1):
        self.gate_vision_data = VisionData()
        self.gate_vision_data.align_target_center_x = None
        self.gate_vision_data.gate_is_visible = False

        # --- MODIFIED: Added Stabilize after AlignToObjectX ---
        self.subtasks = [
            WaitForTargetVisible(), # Wait to see the gate
            AlignToObjectX(target_x_fraction=0.5,
                           tolerance_px=10,
                           yaw_gain=1.5, # Keep increased gain
                           yaw_rate_tolerance=0.05), # Align center
            Stabilize(duration=1.0, speed_threshold=0.1), # Stabilize briefly after aligning
            DriveUntilTargetLost(surge_power=0.6), # Drive forward WHILE gate is visible
            DriveStraight(duration=4.0, surge_power=0.6), # Drive straight AFTER losing sight
            Stabilize(duration=1.0) # Optional: Stabilize after fully passing
        ]
        # ---

        super().__init__()
        self.reset()

    # (process_vision method remains the same)
    def process_vision(self, sub: 'Submarine', camera_image: pygame.Surface) -> VisionData:
        # --- MODIFICATION: Use RED_HSV_RANGES and update variable name ---
        red_blobs = find_blobs_hsv(camera_image, RED_HSV_RANGES, 50) # Was GRAY_HSV_RANGE
        self.gate_vision_data.gate_is_visible = False
        self.gate_vision_data.align_target_center_x = None
        if len(red_blobs) >= 2: # Was gray_blobs
            potential_poles = [b for b in red_blobs if b['height'] > b['width'] * 1.5] # Was gray_blobs
            # ---
            if len(potential_poles) >= 2:
                potential_poles.sort(key=lambda p: p['center_x'])
                best_pair, min_height_diff = None, float('inf')
                for i in range(len(potential_poles) - 1):
                    p1, p2 = potential_poles[i], potential_poles[i+1]
                    height_diff = abs(p1['height'] - p2['height'])
                    y_overlap = min(p1['max_y'], p2['max_y']) - max(p1['min_y'], p2['min_y'])
                    if y_overlap > 10 and height_diff < min_height_diff:
                         min_height_diff = height_diff
                         best_pair = (p1, p2)
                if best_pair is not None and min_height_diff < 150:
                    left_pole, right_pole = best_pair
                    self.gate_vision_data.gate_is_visible = True
                    center_x = (left_pole['center_x'] + right_pole['center_x']) / 2
                    self.gate_vision_data.align_target_center_x = center_x
                    self.gate_vision_data.min_x = left_pole['min_x']
                    self.gate_vision_data.max_x = right_pole['max_x']
        return self.gate_vision_data

    # Execute method is inherited
