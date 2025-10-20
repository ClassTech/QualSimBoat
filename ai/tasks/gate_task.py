#!/usr/bin/env python3
"""
Contains the specific implementation for the Gate task.
"""
import math
from enum import Enum, auto
from typing import Tuple
import pygame
import numpy as np

from .task_base import Task, TaskStatus
from data_structures import SensorSuite, VisionData, ThrusterCommands
from config import SimulationConfig, GRAY_HSV_RANGE
from ai.vision import find_blobs_hsv
from utils import angle_diff


class GateTaskState(Enum):
    SEARCHING = auto()
    ALIGNING = auto()
    APPROACHING = auto()
    CLEARING_GATE = auto()

class GateTask(Task):
    def __init__(self, target_depth: float):
        self.target_depth = target_depth # This is still used for the AI's internal state
        self.ALIGN_CENTER_TOLERANCE_PX = 10 
        self.ALIGN_SQUARE_TOLERANCE_PX = 10 
        self.ALIGN_YAW_RATE_TOLERANCE_RPS = 0.05 
        self.CLEAR_GATE_DURATION = 3.5
        self.reset()

    def reset(self):
        self.current_state = GateTaskState.SEARCHING
        self.state_timer = 0.0
        self.search_start_heading, self.has_completed_spin = None, False
        self.time_since_gate_lost = 0.0

    @property
    def state_name(self) -> str:
        return self.current_state.name

    def process_vision(self, sub: 'Submarine', camera_image: pygame.Surface) -> VisionData:
        vision_data = VisionData()
        
        gray_blobs = find_blobs_hsv(camera_image, GRAY_HSV_RANGE, sub.MIN_PIXELS_FOR_DETECTION)
        
        if not gray_blobs or len(gray_blobs) < 2:
            return vision_data

        potential_poles = [b for b in gray_blobs if b['height'] > b['width'] * 2.0]

        if len(potential_poles) < 2:
            return vision_data
        
        potential_poles.sort(key=lambda p: p['center_x'])
        
        best_pair, min_height_diff = None, float('inf')
        
        for i in range(len(potential_poles) - 1):
            p1, p2 = potential_poles[i], potential_poles[i+1]
            height_diff = abs(p1['height'] - p2['height'])
            
            if height_diff < min_height_diff:
                min_height_diff = height_diff
                best_pair = (p1, p2)
        
        if best_pair is None or min_height_diff > 100:
             return vision_data

        left_pole, right_pole = best_pair
        
        vision_data.gate_is_visible = True
        vision_data.min_x = left_pole['min_x']
        vision_data.max_x = right_pole['max_x']
        vision_data.min_y = min(left_pole['min_y'], right_pole['min_y'])
        vision_data.max_y = max(left_pole['max_y'], right_pole['max_y'])
        vision_data.apparent_height = (left_pole['height'] + right_pole['height']) / 2
        vision_data.gate_center_y = (left_pole['center_y'] + right_pole['center_y']) / 2
        
        gate_center_x = (left_pole['center_x'] + right_pole['center_x']) / 2
        vision_data.left_passage_center_x = gate_center_x
        vision_data.right_passage_center_x = gate_center_x

        return vision_data

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: SimulationConfig) -> Tuple[TaskStatus, ThrusterCommands]:
        if self.current_state == GateTaskState.SEARCHING:
            if vision_data.gate_is_visible:
                self.current_state = GateTaskState.ALIGNING
                return TaskStatus.RUNNING, ThrusterCommands() 
            else:
                if self.search_start_heading is None: 
                    self.search_start_heading, self.has_completed_spin = sensors.heading, False
                if not self.has_completed_spin and abs(angle_diff(sensors.heading, self.search_start_heading)) < 15.0 and self.state_timer > 2.0: 
                    self.has_completed_spin = True
                
                self.state_timer += dt
                return TaskStatus.RUNNING, sub.get_search_commands(sensors)
        
        if self.current_state == GateTaskState.ALIGNING:
            if not vision_data.gate_is_visible:
                self.time_since_gate_lost += dt
                if self.time_since_gate_lost > 2.0:
                    self.current_state = GateTaskState.SEARCHING
                    return TaskStatus.RUNNING, ThrusterCommands()
                else:
                    return TaskStatus.RUNNING, sub.get_spin_damping_commands(sensors)
            
            self.time_since_gate_lost = 0.0
            cam_w, cam_h = sensors.camera_image.get_size()
            gate_center_x = (vision_data.min_x + vision_data.max_x) / 2
            
            # 1. Yaw control
            pixel_error_x = gate_center_x - (cam_w / 2)
            yaw_p = -(pixel_error_x / (cam_w / 2)) * sub.ALIGN_YAW_P_GAIN
            yaw_d = -sensors.imu.gyro_z * sub.YAW_D_GAIN
            yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)

            surge = 0.0
            sway = 0.0 
            
            # 4. Completion condition
            is_centered = abs(pixel_error_x) < self.ALIGN_CENTER_TOLERANCE_PX
            is_stable_rotation = abs(sensors.imu.gyro_z) < self.ALIGN_YAW_RATE_TOLERANCE_RPS
            
            if is_centered and is_stable_rotation:
                sub.approach_heading = sensors.heading
                self.current_state = GateTaskState.APPROACHING
                return TaskStatus.RUNNING, sub._get_damping_commands(sensors)

            return TaskStatus.RUNNING, sub._mix_and_normalize_commands(surge, sway, yaw)
        
        if self.current_state == GateTaskState.APPROACHING:
            if vision_data.gate_is_visible:
                self.time_since_gate_lost = 0.0
            else:
                self.time_since_gate_lost += dt

            if self.time_since_gate_lost > 0.75:
                self.current_state = GateTaskState.CLEARING_GATE
                self.state_timer = self.CLEAR_GATE_DURATION
                sub.pass_start_pos = (sensors.x, sensors.y)
                return TaskStatus.RUNNING, ThrusterCommands()

            nav_target_x, side = sub._get_navigation_target(vision_data, sensors.camera_image.get_width())
            sub.gate_passage_side = side
            if nav_target_x is None:
                nav_target_x = sensors.camera_image.get_width() / 2
            
            return TaskStatus.RUNNING, sub.get_go_to_visual_target_commands(sensors, nav_target_x, sub.SURGE_MAX_SPEED)

        if self.current_state == GateTaskState.CLEARING_GATE:
            self.state_timer -= dt
            if self.state_timer <= 0:
                self.gateCompleted = True
                sub.target_x, sub.target_y = sensors.x, sensors.y
                sub.target_heading = sensors.heading
                # --- THIS IS THE FIX ---
                return TaskStatus.COMPLETED, sub._get_damping_commands(sensors)
                # --- END FIX ---
            
            return TaskStatus.RUNNING, sub.get_heading_commands(sensors, sensors.heading, sub.SURGE_MAX_SPEED)
        
        return TaskStatus.RUNNING, ThrusterCommands()