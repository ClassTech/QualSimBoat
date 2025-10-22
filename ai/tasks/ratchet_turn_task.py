#!/usr/bin/env python3
"""
Implements a "Ratchet Turn" task around the green marker pole.
Aligns to an initial fraction, surges past, yaws back, then aligns
to a subsequent fraction, repeats until the gate is seen.
"""
import math
from enum import Enum, auto
from typing import Tuple, Dict, Any
import pygame
import numpy as np

# Absolute imports
from ai.tasks.task_base import Task, TaskStatus
from ai.tasks.subtask_base import SubtaskStatus # Only needed for status comparison
# No longer using AlignToObjectX logic directly
# from ai.tasks.common_subtasks import AlignToObjectX 

from data_structures import SensorSuite, VisionData, ThrusterCommands
from config import SimulationConfig, GREEN_HSV_RANGE, RED_HSV_RANGES # Import both color ranges
from ai.vision import find_blobs_hsv
from utils import angle_diff

# Define internal states for the task's state machine
class RatchetState(Enum):
    FIND_POLE = auto()
    ALIGN_POLE = auto()
    SURGE_PAST_POLE = auto()
    YAW_BACK_TO_POLE = auto()
    GATE_FOUND = auto() # Terminal state for completion

class RatchetTurnTask(Task):
    """
    Navigates around the pole using a look-surge-yaw-repeat strategy
    until the red gate is detected. Uses different alignment targets.
    """
    def __init__(self, 
                 initial_target_fraction=0.65, 
                 subsequent_target_fraction=0.95, # New parameter
                 surge_power=0.3, 
                 yaw_power=-0.3): # Default yaw left
        # Task parameters
        self.initial_target_fraction = initial_target_fraction
        self.subsequent_target_fraction = subsequent_target_fraction # New
        self.surge_power = surge_power
        self.yaw_power = yaw_power 
        self.align_tolerance_px = 15
        self.align_yaw_gain = 1.5 
        self.disappear_threshold_fraction = 0.95 
        self.align_yaw_rate_tolerance = 0.05 

        # Internal state variables
        self.state = RatchetState.FIND_POLE
        self.heading_to_hold = None
        self.last_pole_center_x = None 
        self.is_initial_alignment = True # New flag

        # Vision data storage
        self.pole_vision_data = VisionData()
        self.gate_vision_data = VisionData()

        self.subtasks = [] 
        super().__init__() 
        self.reset() 

    # Override reset for internal state
    def reset(self, search_direction: int = 1): 
        self.state = RatchetState.FIND_POLE
        self.heading_to_hold = None
        self.last_pole_center_x = None
        self.is_initial_alignment = True # Reset flag
        self.pole_vision_data = VisionData()
        self.gate_vision_data = VisionData()
        print("RatchetTurnTask Reset") 

    @property
    def state_name(self) -> str:
        # Include whether it's initial alignment
        align_target = self.initial_target_fraction if self.is_initial_alignment else self.subsequent_target_fraction
        return f"RatchetTurn:{self.state.name}({align_target*100:.0f}%)"

    # Override process_vision (remains the same as previous version)
    def process_vision(self, sub: 'Submarine', camera_image: pygame.Surface) -> VisionData:
        # --- Process for GREEN POLE ---
        green_blobs = find_blobs_hsv(camera_image, GREEN_HSV_RANGE, sub.MIN_PIXELS_FOR_DETECTION)
        self.pole_vision_data.gate_is_visible = False
        self.pole_vision_data.align_target_center_x = None
        self.pole_vision_data.align_target_width = None
        if green_blobs:
            potential_poles = [b for b in green_blobs if b['height'] > b['width'] * 1.2]
            if potential_poles:
                marker_pole = max(potential_poles, key=lambda b: b['area'])
                self.pole_vision_data.gate_is_visible = True
                self.pole_vision_data.align_target_center_x = marker_pole['center_x']
                self.pole_vision_data.min_x = marker_pole['min_x']
                self.pole_vision_data.max_x = marker_pole['max_x']
                self.pole_vision_data.apparent_height = marker_pole['height']

        # --- Process for RED GATE ---
        red_blobs = find_blobs_hsv(camera_image, RED_HSV_RANGES, 50) 
        self.gate_vision_data.gate_is_visible = False
        if len(red_blobs) >= 2:
            potential_poles = [b for b in red_blobs if b['height'] > b['width'] * 1.5]
            if len(potential_poles) >= 2:
                 self.gate_vision_data.gate_is_visible = True

        return self.pole_vision_data

    # Override execute to implement the modified state machine
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data_ignored: VisionData, config: SimulationConfig) -> Tuple[TaskStatus, ThrusterCommands]:
        
        current_pole_data = self.process_vision(sub, sensors.camera_image)
        cam_w, _ = sensors.camera_image.get_size()
        
        # --- Determine current alignment target based on flag ---
        current_target_fraction = self.initial_target_fraction if self.is_initial_alignment else self.subsequent_target_fraction
        target_pixel_x = cam_w * current_target_fraction
        # Also need the target pixel for the *subsequent* turns during the YAW_BACK state
        subsequent_target_pixel_x = cam_w * self.subsequent_target_fraction
        # ---

        # Check for overall task completion (gate visible)
        if self.state not in [RatchetState.FIND_POLE, RatchetState.GATE_FOUND] and self.gate_vision_data.gate_is_visible:
            print("INFO: RatchetTurnTask - Gate Found! Completing task.")
            self.state = RatchetState.GATE_FOUND
            return TaskStatus.COMPLETED, sub._get_damping_commands(sensors)

        commands = ThrusterCommands() 

        # === State: FIND_POLE ===
        if self.state == RatchetState.FIND_POLE:
            if current_pole_data.gate_is_visible: 
                print("INFO: Ratchet Pole Found. Aligning (Initial).")
                self.state = RatchetState.ALIGN_POLE
            else:
                commands = sub._mix_and_normalize_commands(0.0, 0.0, self.yaw_power)

        # === State: ALIGN_POLE ===
        elif self.state == RatchetState.ALIGN_POLE:
            if not current_pole_data.gate_is_visible or current_pole_data.align_target_center_x is None:
                 print("ERROR: Ratchet - Lost pole during alignment.")
                 return TaskStatus.FAILED, sub.get_spin_damping_commands(sensors)

            current_center_x = current_pole_data.align_target_center_x
            # --- Use the correct target pixel x for error calculation ---
            pixel_error_x = current_center_x - target_pixel_x 
            # ---

            is_centered = abs(pixel_error_x) < self.align_tolerance_px
            is_stable = abs(sensors.imu.gyro_z) < self.align_yaw_rate_tolerance
           
            if is_centered and is_stable:
                align_type = "Initial" if self.is_initial_alignment else "Subsequent"
                print(f"INFO: Ratchet Aligned ({align_type}). Surging past.")
                self.heading_to_hold = sensors.heading 
                self.last_pole_center_x = current_center_x 
                self.state = RatchetState.SURGE_PAST_POLE
                # --- Set flag to False after the first alignment is done ---
                if self.is_initial_alignment:
                    self.is_initial_alignment = False
                # ---
                commands = sub.get_heading_commands(sensors, self.heading_to_hold, self.surge_power) 
            else:
                # Calculate alignment yaw command
                yaw_p = -(pixel_error_x / (cam_w / 2)) * self.align_yaw_gain
                yaw_d = -sensors.imu.gyro_z * sub.YAW_D_GAIN 
                yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)
                commands = sub._mix_and_normalize_commands(0.0, 0.0, yaw) 

        # === State: SURGE_PAST_POLE ===
        elif self.state == RatchetState.SURGE_PAST_POLE:
            if self.heading_to_hold is None: 
                 self.heading_to_hold = sensors.heading

            pole_visible = current_pole_data.gate_is_visible
            pole_center_x = current_pole_data.align_target_center_x

            pole_disappeared_right = (not pole_visible) and \
                                     (self.last_pole_center_x is not None) and \
                                     (self.last_pole_center_x > cam_w * self.disappear_threshold_fraction)

            if pole_disappeared_right:
                print("INFO: Ratchet Pole Passed Right Edge. Yawing back.")
                self.state = RatchetState.YAW_BACK_TO_POLE
                commands = sub._mix_and_normalize_commands(0.0, 0.0, self.yaw_power) 
            else:
                commands = sub.get_heading_commands(sensors, self.heading_to_hold, self.surge_power)
                if pole_visible and pole_center_x is not None:
                    self.last_pole_center_x = pole_center_x
                else:
                    self.last_pole_center_x = None 

        # === State: YAW_BACK_TO_POLE ===
        elif self.state == RatchetState.YAW_BACK_TO_POLE:
            pole_visible = current_pole_data.gate_is_visible
            pole_center_x = current_pole_data.align_target_center_x

            # --- Check against the SUBSEQUENT target pixel x ---
            if pole_visible and pole_center_x is not None and pole_center_x < subsequent_target_pixel_x:
            # ---    
                print("INFO: Ratchet Pole Reacquired Left of Target. Re-aligning (Subsequent).")
                self.state = RatchetState.ALIGN_POLE 
            else:
                commands = sub._mix_and_normalize_commands(0.0, 0.0, self.yaw_power)

        # === State: GATE_FOUND ===
        elif self.state == RatchetState.GATE_FOUND:
            commands = sub._get_damping_commands(sensors)

        # Return status and commands
        if self.state == RatchetState.GATE_FOUND:
            return TaskStatus.COMPLETED, sub._get_damping_commands(sensors)
        else:
            return TaskStatus.RUNNING, commands