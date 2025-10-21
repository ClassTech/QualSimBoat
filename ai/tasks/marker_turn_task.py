#!/usr/bin/env python3
"""
Refactored Marker Turn task using reusable subtasks and context.
Handles subtask failure by restarting itself with a rightward search.
Orbit subtask includes local search on temporary target loss.
"""
import math
from enum import Enum, auto
from typing import Tuple, List, Dict, Any
import pygame
import numpy as np

# Absolute imports
from ai.tasks.task_base import Task, TaskStatus
from ai.tasks.subtask_base import Subtask, SubtaskStatus
from ai.tasks.common_subtasks import (WaitForTargetVisible, AlignToObjectX, DriveStraight,
                                      TurnToHeading, Stabilize)

from data_structures import SensorSuite, VisionData, ThrusterCommands
from config import SimulationConfig, GREEN_HSV_RANGE
from ai.vision import find_blobs_hsv
from utils import angle_diff


class MarkerTurnTaskState(Enum): # (Enums remain the same)
    SEARCHING = auto(); ALIGN_OFFSET = auto(); ORBIT_POLE = auto(); YAW_HOME = auto()

class MarkerTurnTask(Task):
    """Finds, orbits (using visual offset), and turns around the green marker."""

    # (_StoreHeadingSubtask definition remains the same)
    class _StoreHeadingSubtask(Subtask):
        def __init__(self, parent_task: 'MarkerTurnTask'): pass # No need to store parent ref
        def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: SimulationConfig, context: dict) -> Tuple[SubtaskStatus, ThrusterCommands]:
            context['initial_heading'] = sensors.heading; return SubtaskStatus.COMPLETED, ThrusterCommands()

    # --- MODIFIED: _OrbitPoleSubtask with local search ---
    class _OrbitPoleSubtask(Subtask):
        def __init__(self, parent_task: 'MarkerTurnTask'):
            self.parent = parent_task
            self.orbit_initial_heading = None
            self.time_since_target_lost = 0.0
            self.lost_search_timeout = 3.0 # How long to search locally before failing
            self.local_search_turn_power = -0.25 # Turn RIGHT to reacquire

        def on_enter(self, sub: 'Submarine', sensors: SensorSuite, vision_data: VisionData, context: Dict[str, Any]):
            self.orbit_initial_heading = context.get('initial_heading')
            self.time_since_target_lost = 0.0
            if self.orbit_initial_heading is None:
                print("ERROR: OrbitPoleSubtask entered without 'initial_heading' in context! Using current.")
                self.orbit_initial_heading = sensors.heading
                context['initial_heading'] = sensors.heading

        def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
            if self.orbit_initial_heading is None:
                print("ERROR: OrbitPoleSubtask initial_heading is None during execute!")
                return SubtaskStatus.FAILED, sub.get_spin_damping_commands(sensors)

            heading_change = abs(angle_diff(sensors.heading, self.orbit_initial_heading))
            context['current_heading_change'] = heading_change

            # Check completion angle FIRST
            if heading_change > self.parent.ORBIT_COMPLETION_ANGLE_DEG:
                context['final_turn_target'] = (self.orbit_initial_heading - 180.0) % 360
                return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)

            # --- Target VISIBLE Logic ---
            if vision_data.gate_is_visible: # Check marker visibility flag
                self.time_since_target_lost = 0.0 # Reset lost timer

                cam_w, _ = sensors.camera_image.get_size()
                target_pixel_x = cam_w * self.parent.ORBIT_TARGET_X_FRACTION
                marker_center_x = vision_data.align_target_center_x
                if marker_center_x is None:
                     print("WARN: Orbiting, target visible but align_target_center_x is None!")
                     # Continue surging, damp yaw
                     yaw = -sensors.imu.gyro_z * sub.YAW_D_GAIN
                     return TaskStatus.RUNNING, sub._mix_and_normalize_commands(self.parent.ORBIT_SURGE_POWER, 0.0, yaw)

                pixel_error_x = marker_center_x - target_pixel_x
                yaw_p = -(pixel_error_x / (cam_w / 2)) * self.parent.ORBIT_YAW_GAIN
                yaw_d = -sensors.imu.gyro_z * sub.YAW_D_GAIN
                yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)
                # print(f"DEBUG: Orbiting: HdgChg={heading_change:.1f}, PxErr={pixel_error_x:.1f}, Yaw={yaw:.2f}") # DEBUG
                return TaskStatus.RUNNING, sub._mix_and_normalize_commands(self.parent.ORBIT_SURGE_POWER, 0.0, yaw)

            # --- Target LOST Logic ---
            else:
                self.time_since_target_lost += dt
                if self.time_since_target_lost > self.lost_search_timeout:
                   # Failed to reacquire locally, fail the subtask
                   print(f"WARN: OrbitPole lost target for > {self.lost_search_timeout}s, failing subtask.")
                   return SubtaskStatus.FAILED, sub.get_spin_damping_commands(sensors)
                else:
                   # Execute local search (turn right, zero surge)
                   # print(f"DEBUG: OrbitPole lost target, searching right ({self.time_since_target_lost:.1f}s)") # DEBUG
                   yaw = self.local_search_turn_power
                   return TaskStatus.RUNNING, sub._mix_and_normalize_commands(0.0, 0.0, yaw)

        # Dynamic name for UI
        def get_dynamic_name(self, context: Dict[str, Any]) -> str:
            ch = context.get('current_heading_change', 0.0); ca = self.parent.ORBIT_COMPLETION_ANGLE_DEG
            # Add indicator if searching locally
            search_indicator = " Searching!" if self.time_since_target_lost > 0 else ""
            return f"OrbitPole({ch:.0f}/{ca:.0f}){search_indicator}"
    # --- End _OrbitPoleSubtask ---

    def __init__(self, target_depth: float = 0.1):
        # (Initialization parameters remain the same)
        self.marker_vision_data = VisionData(); self.marker_vision_data.align_target_center_x = None; self.marker_vision_data.align_target_width = None; self.marker_vision_data.gate_is_visible = False
        # self.initial_heading = None # Managed via context
        self.TURN_COMPLETION_TOLERANCE_DEG = 10.0; self.ALIGN_CENTER_TOLERANCE_PX = 20; self.ALIGN_YAW_RATE_TOLERANCE_RPS = 0.05
        self.ORBIT_TARGET_X_FRACTION = 0.60; self.ORBIT_YAW_GAIN = 3.0; self.ORBIT_SURGE_POWER = 0.3; self.ORBIT_COMPLETION_ANGLE_DEG = 110.0

        # (Subtask list remains the same)
        self.subtasks = [
            WaitForTargetVisible(),
            AlignToObjectX(target_x_fraction=self.ORBIT_TARGET_X_FRACTION, tolerance_px=self.ALIGN_CENTER_TOLERANCE_PX, yaw_gain=self.ORBIT_YAW_GAIN, yaw_rate_tolerance=self.ALIGN_YAW_RATE_TOLERANCE_RPS),
            MarkerTurnTask._StoreHeadingSubtask(self),
            MarkerTurnTask._OrbitPoleSubtask(self),
            TurnToHeading(relative_degrees=-180.0, tolerance_degrees=self.TURN_COMPLETION_TOLERANCE_DEG), # Yaw Home
        ]
        self.yaw_home_subtask = self.subtasks[4] # Index adjusted
        super().__init__(); self.reset()

    # (process_vision remains the same)
    def process_vision(self, sub: 'Submarine', camera_image: pygame.Surface) -> VisionData: # ... as before ...
        green_blobs = find_blobs_hsv(camera_image, GREEN_HSV_RANGE, sub.MIN_PIXELS_FOR_DETECTION)
        self.marker_vision_data.gate_is_visible = False; self.marker_vision_data.align_target_center_x = None; self.marker_vision_data.align_target_width = None
        if not green_blobs: return self.marker_vision_data
        potential_poles = [b for b in green_blobs if b['height'] > b['width'] * 1.2]
        if not potential_poles: return self.marker_vision_data
        marker_pole = max(potential_poles, key=lambda b: b['area'])
        self.marker_vision_data.gate_is_visible = True; self.marker_vision_data.align_target_center_x = marker_pole['center_x']
        self.marker_vision_data.align_target_width = marker_pole['width']; self.marker_vision_data.min_x = marker_pole['min_x']; self.marker_vision_data.max_x = marker_pole['max_x']
        return self.marker_vision_data

    # (execute override for handling failure remains the same)
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data_ignored: VisionData, config: SimulationConfig) -> Tuple[TaskStatus, ThrusterCommands]: # ... as before ...
         processed_vision_data = self.process_vision(sub, sensors.camera_image)
         # Dynamic turn target setting logic for Yaw Home
         if self.subtasks and 0 <= self.current_subtask_index < len(self.subtasks):
             current_subtask = self.subtasks[self.current_subtask_index]
             if current_subtask is self.yaw_home_subtask and self.yaw_home_subtask.target_heading is None:
                  target = self.context.get('final_turn_target')
                  if target is not None: self.yaw_home_subtask.set_target(target)
                  else: # Fallback
                       init_h = self.context.get('initial_heading')
                       if init_h is not None: self.yaw_home_subtask.set_target((init_h - 180.0) % 360)
                       else: print("ERROR: Yaw Home cannot determine target!"); return TaskStatus.FAILED, sub._get_damping_commands(sensors)

         # Call base class execute
         status, commands = super().execute(sub, dt, sensors, processed_vision_data, config)

         # Handle FAILED status from base execute by restarting this task
         if status == TaskStatus.FAILED:
             print(f"INFO: {self.__class__.__name__} caught subtask failure. Resetting.")
             self.reset(search_direction=-1) # Restart with right search
             return TaskStatus.RUNNING, sub._get_damping_commands(sensors)

         return status, commands