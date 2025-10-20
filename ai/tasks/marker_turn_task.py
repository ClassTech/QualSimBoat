#!/usr/bin/env python3
"""
Contains the specific implementation for the Marker Turn task using an orbiting maneuver.
"""
import math
from enum import Enum, auto
from typing import Tuple
import pygame
import numpy as np

from .task_base import Task, TaskStatus
from data_structures import SensorSuite, VisionData, ThrusterCommands
from config import SimulationConfig, GREEN_HSV_RANGE
from ai.vision import find_blobs_hsv
from utils import angle_diff

class MarkerTurnTaskState(Enum):
    SEARCHING = auto()
    ALIGN_OFFSET = auto()
    ORBIT_POLE = auto()
    YAW_HOME = auto()

class MarkerTurnTask(Task):
    def __init__(self, target_depth: float):
        self.target_depth = target_depth
        self.TURN_COMPLETION_TOLERANCE_DEG = 10.0
        self.ALIGN_CENTER_TOLERANCE_PX = 20
        self.ALIGN_YAW_RATE_TOLERANCE_RPS = 0.05
        # --- Orbit parameters ---
        self.ORBIT_TARGET_X_FRACTION = 0.75
        # --- MODIFIED: Further Increased Yaw Gain ---
        self.ORBIT_YAW_GAIN = 2.5          # Was 1.8 (Even more aggressive correction)
        # ---
        self.ORBIT_SURGE_POWER = 0.35
        self.ORBIT_COMPLETION_ANGLE_DEG = 110.0
        self.reset()

    def reset(self):
        self.current_state = MarkerTurnTaskState.SEARCHING
        self.target_heading = None
        self.time_since_target_lost = 0.0
        self.orbit_initial_heading = None

    @property
    def state_name(self) -> str:
        if self.current_state == MarkerTurnTaskState.ORBIT_POLE and hasattr(self, '_current_heading_change'):
            return f"ORBIT_POLE ({self._current_heading_change:.1f}°/{self.ORBIT_COMPLETION_ANGLE_DEG:.1f}°)"
        return self.current_state.name

    def process_vision(self, sub: 'Submarine', camera_image: pygame.Surface) -> VisionData:
        # (Vision processing remains the same)
        vision_data = VisionData()
        green_blobs = find_blobs_hsv(camera_image, GREEN_HSV_RANGE, sub.MIN_PIXELS_FOR_DETECTION)
        if not green_blobs: return vision_data
        potential_poles = [b for b in green_blobs if b['height'] > b['width'] * 1.5]
        if not potential_poles: return vision_data
        marker_pole = max(potential_poles, key=lambda b: b['area'])
        vision_data.gate_is_visible = True
        vision_data.min_x, vision_data.max_x = marker_pole['min_x'], marker_pole['max_x']
        vision_data.min_y, vision_data.max_y = marker_pole['min_y'], marker_pole['max_y']
        vision_data.apparent_width = marker_pole['width']
        vision_data.left_passage_center_x = marker_pole['center_x']
        vision_data.right_passage_center_x = marker_pole['center_x']
        return vision_data

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: SimulationConfig) -> Tuple[TaskStatus, ThrusterCommands]:

        # --- State 1: SEARCHING ---
        if self.current_state == MarkerTurnTaskState.SEARCHING:
            if vision_data.gate_is_visible:
                self.current_state = MarkerTurnTaskState.ALIGN_OFFSET
                return TaskStatus.RUNNING, ThrusterCommands()
            else:
                return TaskStatus.RUNNING, sub.get_search_commands(sensors)

        # --- State 2: ALIGN_OFFSET ---
        if self.current_state == MarkerTurnTaskState.ALIGN_OFFSET:
            if not vision_data.gate_is_visible:
                self.time_since_target_lost += dt
                if self.time_since_target_lost > 2.0:
                    self.current_state = MarkerTurnTaskState.SEARCHING
                return TaskStatus.RUNNING, sub.get_spin_damping_commands(sensors)

            self.time_since_target_lost = 0.0
            cam_w, _ = sensors.camera_image.get_size()
            target_pixel_x = cam_w * self.ORBIT_TARGET_X_FRACTION
            marker_center_x = vision_data.left_passage_center_x
            pixel_error_x = marker_center_x - target_pixel_x
            yaw_p = -(pixel_error_x / (cam_w / 2)) * self.ORBIT_YAW_GAIN # Use aggressive gain here too
            yaw_d = -sensors.imu.gyro_z * sub.YAW_D_GAIN
            yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)
            is_centered_offset = abs(pixel_error_x) < self.ALIGN_CENTER_TOLERANCE_PX
            is_stable = abs(sensors.imu.gyro_z) < self.ALIGN_YAW_RATE_TOLERANCE_RPS

            if is_centered_offset and is_stable:
                self.current_state = MarkerTurnTaskState.ORBIT_POLE
                self.orbit_initial_heading = sensors.heading
                return TaskStatus.RUNNING, ThrusterCommands()

            return TaskStatus.RUNNING, sub._mix_and_normalize_commands(0.0, 0.0, yaw) # Pivot

        # --- State 3: ORBIT_POLE ---
        if self.current_state == MarkerTurnTaskState.ORBIT_POLE:
            if self.orbit_initial_heading is None:
                self.orbit_initial_heading = sensors.heading

            heading_change = abs(angle_diff(sensors.heading, self.orbit_initial_heading))
            self._current_heading_change = heading_change

            # --- MODIFIED: Check heading completion FIRST ---
            # If we've turned enough, transition regardless of current visibility
            if heading_change > self.ORBIT_COMPLETION_ANGLE_DEG:
                self.current_state = MarkerTurnTaskState.YAW_HOME
                self.target_heading = (self.orbit_initial_heading - 180.0) % 360
                # Damp motion before the final turn
                return TaskStatus.RUNNING, sub._get_damping_commands(sensors)
            # --- END MODIFICATION ---

            # If heading condition not met, THEN check visibility and control orbit
            if not vision_data.gate_is_visible:
                self.time_since_target_lost += dt
                if self.time_since_target_lost > 1.5: # Timeout if lost too long
                   self.current_state = MarkerTurnTaskState.SEARCHING
                   return TaskStatus.RUNNING, sub.get_spin_damping_commands(sensors)
                # If lost briefly, just damp yaw and keep surging
                yaw = -sensors.imu.gyro_z * sub.YAW_D_GAIN
                return TaskStatus.RUNNING, sub._mix_and_normalize_commands(self.ORBIT_SURGE_POWER, 0.0, yaw)

            # --- If visible and not done turning ---
            self.time_since_target_lost = 0.0
            cam_w, _ = sensors.camera_image.get_size()
            target_pixel_x = cam_w * self.ORBIT_TARGET_X_FRACTION
            marker_center_x = vision_data.left_passage_center_x
            pixel_error_x = marker_center_x - target_pixel_x

            # Use aggressive yaw control
            yaw_p = -(pixel_error_x / (cam_w / 2)) * self.ORBIT_YAW_GAIN
            yaw_d = -sensors.imu.gyro_z * sub.YAW_D_GAIN
            yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)

            # Command orbit surge and corrective yaw
            return TaskStatus.RUNNING, sub._mix_and_normalize_commands(self.ORBIT_SURGE_POWER, 0.0, yaw)

        # --- State 4: YAW_HOME ---
        if self.current_state == MarkerTurnTaskState.YAW_HOME:
            if self.target_heading is None:
                 initial_heading = self.orbit_initial_heading if self.orbit_initial_heading is not None else sensors.heading
                 self.target_heading = (initial_heading - 180.0) % 360

            heading_error = angle_diff(self.target_heading, sensors.heading)
            if abs(heading_error) < self.TURN_COMPLETION_TOLERANCE_DEG:
                return TaskStatus.COMPLETED, sub._get_damping_commands(sensors)

            return TaskStatus.RUNNING, sub.get_heading_commands(sensors, self.target_heading, surge_power=0.0) # Pivot

        return TaskStatus.RUNNING, ThrusterCommands()