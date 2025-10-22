#!/usr/bin/env python3
"""
Refactored Marker Turn task using reusable subtasks and context.
Handles subtask failure by restarting itself with a rightward search.
Implements a dynamic orbit subtask. Uses the Vision class.
"""
import math
from enum import Enum, auto
from typing import Tuple, List, Dict, Any
import pygame
import numpy as np

# Absolute imports
from ai.tasks.task_base import Task, TaskStatus
from ai.tasks.subtask_base import Subtask, SubtaskStatus
# Import necessary common subtasks
from ai.tasks.common_subtasks import (WaitForTargetVisible, AlignToObjectX, 
                                      TurnToHeading) # Removed unused imports

# --- Import Vision class ---
from data_structures import SensorSuite, Vision, ThrusterCommands 
# ---
from config import SimulationConfig # Keep config for subtask calls
# Vision functions/config no longer needed here
# from config import GREEN_HSV_RANGE
# from ai.vision import find_blobs_hsv 
# ---
from utils import angle_diff


class MarkerTurnTaskState(Enum): # Unused now, but harmless
    SEARCHING = auto(); ALIGN_OFFSET = auto(); ORBIT_POLE = auto(); YAW_HOME = auto()

class MarkerTurnTask(Task):
    """
    Finds, orbits (using visual offset), and turns around the green marker.
    Uses the Vision class passed from Submarine.
    """

    # --- UPDATED _StoreHeadingSubtask ---
    class _StoreHeadingSubtask(Subtask):
        def __init__(self, parent_task: 'MarkerTurnTask'): pass 
        # --- Update type hint ---
        def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision: Vision, config: SimulationConfig, context: dict) -> Tuple[SubtaskStatus, ThrusterCommands]:
        # ---
            context['initial_heading'] = sensors.heading
            return SubtaskStatus.COMPLETED, ThrusterCommands()
    # ---

    # --- UPDATED _DynamicOrbitSubtask ---
    class _DynamicOrbitSubtask(Subtask):
        """
        Orbits the pole by applying constant surge and using a dynamic yaw 
        target that moves from a start fraction to an end fraction. Uses Vision object.
        """
        def __init__(self, parent_task: 'MarkerTurnTask'):
            self.parent = parent_task
            self.orbit_initial_heading = None
            self.time_since_target_lost = 0.0
            self.lost_search_timeout = 3.0 
            self.local_search_turn_power = -0.25 

        # --- Update type hint ---
        def on_enter(self, sub: 'Submarine', sensors: SensorSuite, vision: Vision, context: Dict[str, Any]):
        # ---
            self.time_since_target_lost = 0.0
            self.orbit_initial_heading = context.get('initial_heading')
            if self.orbit_initial_heading is None:
                print("ERROR: DynamicOrbitSubtask entered without 'initial_heading' in context! Using current.")
                self.orbit_initial_heading = sensors.heading
                context['initial_heading'] = sensors.heading

        # --- Update type hint and logic ---
        def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision: Vision, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        # ---
            if self.orbit_initial_heading is None:
                print("ERROR: DynamicOrbitSubtask initial_heading is None during execute!")
                return SubtaskStatus.FAILED, sub.get_spin_damping_commands(sensors)

            heading_change = abs(angle_diff(sensors.heading, self.orbit_initial_heading))
            context['current_heading_change'] = heading_change

            if heading_change > self.parent.ORBIT_COMPLETION_ANGLE_DEG:
                context['final_turn_target'] = (self.orbit_initial_heading - 180.0) % 360
                return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)

            # --- Use Vision methods ---
            pole_visible = vision.is_pole_visible()
            # ---

            if pole_visible: 
                self.time_since_target_lost = 0.0 
                cam_w, _ = sensors.camera_image.get_size()
                
                # DYNAMIC YAW CONTROL 
                progress = min(1.0, heading_change / self.parent.ORBIT_COMPLETION_ANGLE_DEG)
                start_frac = self.parent.ORBIT_START_X_FRACTION
                end_frac = self.parent.ORBIT_END_X_FRACTION
                target_frac = start_frac + progress * (end_frac - start_frac)
                target_pixel_x = cam_w * target_frac
                
                # --- Use Vision methods ---
                marker_center_x = vision.get_pole_center_x()
                # ---
                if marker_center_x is None: # Should not happen if is_pole_visible is true, but safety check
                     print("WARN: Orbiting, pole visible but get_pole_center_x is None!")
                     return SubtaskStatus.RUNNING, sub.get_spin_damping_commands(sensors)
                
                pixel_error_x = marker_center_x - target_pixel_x
                yaw_p = -(pixel_error_x / (cam_w / 2)) * self.parent.ORBIT_YAW_GAIN
                yaw_d = -sensors.imu.gyro_z * sub.YAW_D_GAIN
                yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)
                
                # SURGE CONTROL (Constant) 
                surge = self.parent.ORBIT_SURGE_POWER
                
                return SubtaskStatus.RUNNING, sub._mix_and_normalize_commands(surge, 0.0, yaw)

            else: # Target LOST Logic 
                self.time_since_target_lost += dt
                if self.time_since_target_lost > self.lost_search_timeout:
                   print(f"WARN: DynamicOrbit lost target for > {self.lost_search_timeout}s, failing subtask.")
                   return SubtaskStatus.FAILED, sub.get_spin_damping_commands(sensors)
                else:
                   yaw = self.local_search_turn_power
                   return SubtaskStatus.RUNNING, sub._mix_and_normalize_commands(0.0, 0.0, yaw)

        # Dynamic name for UI (remains the same)
        def get_dynamic_name(self, context: Dict[str, Any]) -> str:
            ch = context.get('current_heading_change', 0.0); ca = self.parent.ORBIT_COMPLETION_ANGLE_DEG
            search_indicator = " Searching!" if self.time_since_target_lost > 0 else ""
            return f"DynamicOrbit({ch:.0f}/{ca:.0f}){search_indicator}"
    # --- End _DynamicOrbitSubtask ---

    def __init__(self, target_depth: float = 0.1):
        # No local vision storage needed
        # self.marker_vision_data = Vision(...) 
        
        # Use parameters from last known good configuration for this task
        self.TURN_COMPLETION_TOLERANCE_DEG = 10.0
        self.ALIGN_CENTER_TOLERANCE_PX = 10 
        self.ALIGN_YAW_RATE_TOLERANCE_RPS = 0.05
        
        self.ORBIT_START_X_FRACTION = 0.7 
        self.ORBIT_END_X_FRACTION = 1.0   
        self.ORBIT_SURGE_POWER = 0.25    
        self.ORBIT_YAW_GAIN = 3.0 # Consider tuning YAW_D_GAIN in Submarine if needed
        self.ORBIT_COMPLETION_ANGLE_DEG = 130.0 # Keep the angle that worked

        super().__init__() # Call base __init__ AFTER defining parameters

        # --- Subtask list remains the same ---
        self.subtasks = [
            # Ensure WaitForTargetVisible checks for 'pole'
            WaitForTargetVisible(target_type='pole'), 
            AlignToObjectX(
                target_x_fraction=self.ORBIT_START_X_FRACTION,
                tolerance_px=self.ALIGN_CENTER_TOLERANCE_PX,
                yaw_gain=self.ORBIT_YAW_GAIN,
                yaw_rate_tolerance=self.ALIGN_YAW_RATE_TOLERANCE_RPS
            ),
            MarkerTurnTask._StoreHeadingSubtask(self),
            MarkerTurnTask._DynamicOrbitSubtask(self),
            TurnToHeading(relative_degrees=-180.0, 
                          tolerance_degrees=self.TURN_COMPLETION_TOLERANCE_DEG),
        ]
        # ---
        
        self.yaw_home_subtask = self.subtasks[4] # Index is 4 now
        self.reset() # Call Task's reset

    # --- process_vision method is REMOVED ---

    # --- UPDATED execute ---
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, 
                # --- vision argument type hint changed ---
                vision: Vision, 
                # ---
                config: SimulationConfig) -> Tuple[TaskStatus, ThrusterCommands]:
        
        # --- vision.update() is called by Submarine BEFORE this ---
        # --- process_vision call is REMOVED ---
        # processed_vision = self.process_vision(sub, sensors.camera_image)
        # --- The vision object passed in is already updated ---
        processed_vision = vision 
        # ---

        # Dynamic turn target setting logic (remains the same)
        if self.subtasks and 0 <= self.current_subtask_index < len(self.subtasks):
             current_subtask = self.subtasks[self.current_subtask_index]
             if current_subtask is self.yaw_home_subtask and self.yaw_home_subtask.target_heading is None:
                  target = self.context.get('final_turn_target')
                  if target is not None: self.yaw_home_subtask.set_target(target)
                  else: 
                       init_h = self.context.get('initial_heading')
                       if init_h is not None: self.yaw_home_subtask.set_target((init_h - 180.0) % 360)
                       else: print("ERROR: Yaw Home cannot determine target!"); return TaskStatus.FAILED, sub._get_damping_commands(sensors)

        # Call base class execute, passing the updated vision object
        status, commands = super().execute(sub, dt, sensors, processed_vision, config)

        # Handle FAILED status from base execute (remains the same)
        if status == TaskStatus.FAILED:
             print(f"INFO: {self.__class__.__name__} caught subtask failure. Resetting.")
             self.reset(search_direction=-1) # Restart with right search
             return TaskStatus.RUNNING, sub._get_damping_commands(sensors)

        return status, commands
    # --- END UPDATED execute ---