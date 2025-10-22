#!/usr/bin/env python3
"""
Implementations of common, reusable Subtasks, using context.
Subtasks now expect a Vision object.
"""
import math
from typing import Tuple, List, Optional, Dict, Any
import numpy as np

# Absolute import for the base class
from ai.tasks.subtask_base import Subtask, SubtaskStatus
# --- Import Vision class ---
from data_structures import SensorSuite, Vision, ThrusterCommands 
# ---
from config import SimulationConfig
from utils import angle_diff

# --- Navigation Subtasks (TurnToHeading, DriveStraight, Stabilize are unchanged) ---
class TurnToHeading(Subtask): # ... as before ...
    def __init__(self, absolute_degrees: Optional[float] = None, relative_degrees: Optional[float] = None, tolerance_degrees: float = 5.0): #...
        if absolute_degrees is not None and relative_degrees is not None: raise ValueError("Provide either absolute_degrees or relative_degrees, not both.")
        if absolute_degrees is None and relative_degrees is None: raise ValueError("Must provide either absolute_degrees or relative_degrees.")
        self.absolute_degrees=absolute_degrees; self.relative_degrees=relative_degrees; self.tolerance=tolerance_degrees; self.target_heading: Optional[float] = None
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite, vision_data: Vision, context: Dict[str, Any]): # vision_data type hint changed
        if self.absolute_degrees is not None: self.target_heading = self.absolute_degrees % 360
        elif self.relative_degrees is not None:
            initial_heading = context.get('initial_heading');
            if initial_heading is None: self.target_heading = None
            else: self.target_heading = (initial_heading + self.relative_degrees) % 360
        else: self.target_heading = None
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: Vision, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]: # vision_data type hint changed
        if self.target_heading is None: print("ERROR: TurnToHeading target could not be determined!"); return SubtaskStatus.FAILED, sub._get_damping_commands(sensors)
        heading_error = angle_diff(self.target_heading, sensors.heading)
        if abs(heading_error) < self.tolerance: return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
        commands = sub.get_heading_commands(sensors, self.target_heading, surge_power=0.0)
        return SubtaskStatus.RUNNING, commands
    def get_dynamic_name(self, context: Dict[str, Any]) -> str: #...
        if self.target_heading is not None: return f"{self.name}({self.target_heading:.0f}°)"
        elif self.absolute_degrees is not None: return f"{self.name}(Abs {self.absolute_degrees:.0f}°)"
        elif self.relative_degrees is not None: rel = self.relative_degrees; init = context.get('initial_heading', '?'); calc_target = f" -> {(init + rel) % 360:.0f}" if init != '?' else ""; return f"{self.name}(Rel {rel:.0f}° from {init}°{calc_target})"
        return self.name

class DriveStraight(Subtask): # ... as before ...
    def __init__(self, duration: float, surge_power: float = 0.5): #...
        self.duration=duration; self.surge_power=surge_power; self.timer=0.0; self.heading_to_hold=None
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite, vision_data: Vision, context: Dict[str, Any]): # vision_data type hint changed
        self.timer = self.duration; self.heading_to_hold = sensors.heading
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: Vision, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]: # vision_data type hint changed
        if self.timer <= 0: return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
        self.timer -= dt
        if self.heading_to_hold is None: self.heading_to_hold = sensors.heading
        commands = sub.get_heading_commands(sensors, self.heading_to_hold, self.surge_power)
        return SubtaskStatus.RUNNING, commands

class Stabilize(Subtask): # ... as before ...
    def __init__(self, duration: float = 2.0, speed_threshold: float = 0.05): #...
        self.duration=duration; self.speed_threshold=speed_threshold; self.timer=0.0; self.target_set=False
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite, vision_data: Vision, context: Dict[str, Any]): # vision_data type hint changed
        self.timer=0.0; self.target_set=False; sub.integral_x_err, sub.integral_y_err = 0.0, 0.0
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: Vision, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]: # vision_data type hint changed
        if not self.target_set: sub.target_x, sub.target_y = sensors.x, sensors.y; sub.target_heading = sensors.heading; self.target_set = True
        self.timer += dt; speed = math.hypot(sensors.velocity_x, sensors.velocity_y)
        if self.timer > self.duration and speed < self.speed_threshold: return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
        commands = sub._get_pid_hover_commands(sensors, dt, sub.target_x, sub.target_y)
        return SubtaskStatus.RUNNING, commands

# --- Vision-Based Subtasks (Updated) ---

class WaitForTargetVisible(Subtask): 
    # Optional: Add parameter to specify waiting for 'pole', 'gate', or 'either'
    def __init__(self, target_type='either'): 
        self.target_type = target_type.lower()
        
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, 
                vision_data: Vision, # Type hint changed
                config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        
        visible = False
        if self.target_type == 'pole':
            visible = vision_data.is_pole_visible()
        elif self.target_type == 'gate':
            visible = vision_data.is_gate_visible()
        else: # 'either'
            visible = vision_data.is_pole_visible() or vision_data.is_gate_visible()
            
        if visible: 
            return SubtaskStatus.COMPLETED, sub.get_spin_damping_commands(sensors)
        else: 
            # Commands are handled by Task base class during search
            return SubtaskStatus.RUNNING, ThrusterCommands()

class AlignToObjectX(Subtask):
    """
    Pivots to center target at target_x_fraction.
    Assumes the Vision object correctly identifies the primary target 
    (pole center for Ratchet, gate center for GateTask) via its methods.
    Stores heading in context['initial_heading'] on completion.
    """
    def __init__(self, target_x_fraction: float, tolerance_px: int = 15, yaw_gain: float = 0.8, yaw_rate_tolerance: float = 0.05):
        self.target_x_fraction=target_x_fraction; self.tolerance_px=tolerance_px; self.yaw_gain=yaw_gain; self.yaw_rate_tolerance=yaw_rate_tolerance

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, 
                vision_data: Vision, # Type hint changed
                config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:

        # --- Get target center from Vision object ---
        # Prioritize gate if visible (for GateTask), else use pole (for RatchetTask)
        current_center_x = None
        if vision_data.is_gate_visible():
            current_center_x = vision_data.get_gate_center_x()
        elif vision_data.is_pole_visible():
            current_center_x = vision_data.get_pole_center_x()
        # ---
            
        if current_center_x is None:
             return SubtaskStatus.RUNNING, ThrusterCommands() 

        cam_w, _ = sensors.camera_image.get_size()
        target_pixel_x = cam_w * self.target_x_fraction
        
        pixel_error_x = current_center_x - target_pixel_x
        yaw_p = -(pixel_error_x / (cam_w / 2)) * self.yaw_gain
        yaw_d = -sensors.imu.gyro_z * sub.YAW_D_GAIN
        yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)

        is_centered = abs(pixel_error_x) < self.tolerance_px
        is_stable = abs(sensors.imu.gyro_z) < self.yaw_rate_tolerance

        if is_centered and is_stable:
             return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)

        return SubtaskStatus.RUNNING, sub._mix_and_normalize_commands(0.0, 0.0, yaw)

    def on_exit(self, sub: 'Submarine', sensors: SensorSuite, 
                vision_data: Vision, # Type hint changed 
                context: Dict[str, Any]):
        context['initial_heading'] = sensors.heading

# Removed ApproachTargetVisualWidth as ApproachAndCenter is preferred now
# class ApproachTargetVisualWidth(...): # ...

# --- ADD DriveUntilTargetLost BACK IN ---
# ... (other imports and subtasks) ...

# --- MODIFIED DriveUntilTargetLost ---
class DriveUntilTargetLost(Subtask): 
    """ 
    Drives straight until the specified vision target (gate or pole) is lost.
    """
    # --- Add target_type ---
    def __init__(self, surge_power: float = 0.5, target_type: str = 'gate'): 
        if target_type not in ['gate', 'pole', 'either']:
             raise ValueError("target_type must be 'gate', 'pole', or 'either'")
        self.surge_power = surge_power
        self.target_type = target_type
        # ---
        self.heading_to_hold: Optional[float] = None
        self.target_was_visible = False # Track if we saw it at least once

    def on_enter(self, sub: 'Submarine', sensors: SensorSuite, 
                 vision: Vision, # Renamed variable
                 context: Dict[str, Any]):
        self.heading_to_hold = context.get('initial_heading', sensors.heading)
        # Check initial visibility based on target_type
        if self.target_type == 'gate':
             self.target_was_visible = vision.is_gate_visible()
        elif self.target_type == 'pole':
             self.target_was_visible = vision.is_pole_visible()
        else: # either
             self.target_was_visible = vision.is_gate_visible() or vision.is_pole_visible()

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, 
                vision: Vision, # Renamed variable
                config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        if self.heading_to_hold is None: self.heading_to_hold = sensors.heading
        
        # --- Check current visibility based on target_type ---
        currently_visible = False
        if self.target_type == 'gate':
             currently_visible = vision.is_gate_visible()
        elif self.target_type == 'pole':
             currently_visible = vision.is_pole_visible()
        else: # either
             currently_visible = vision.is_gate_visible() or vision.is_pole_visible()
        # ---

        if currently_visible:
            self.target_was_visible = True # Mark that we've seen it
            commands = sub.get_heading_commands(sensors, self.heading_to_hold, self.surge_power)
            return SubtaskStatus.RUNNING, commands
        else:
            # Target is not currently visible
            if self.target_was_visible:
                # If we *had* seen it before, but now it's gone, we are done
                return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
            else: 
                # If we never saw it to begin with, something is wrong
                return SubtaskStatus.FAILED, sub._get_damping_commands(sensors)
# --- End Modified ---

# ... (DriveUntilTargetLostRight and other subtasks) ...

class DriveUntilTargetLostRight(Subtask):
    """
    Drives straight, holding heading, until the vision target (assumed pole) 
    disappears off the right side of the screen. Used by RatchetTurnTask.
    """
    def __init__(self, surge_power: float = 0.3, disappear_threshold_fraction: float = 0.95):
        self.surge_power = surge_power
        self.disappear_threshold_fraction = disappear_threshold_fraction
        self.heading_to_hold: Optional[float] = None
        self.last_pole_center_x: Optional[float] = None

    def on_enter(self, sub: 'Submarine', sensors: SensorSuite, 
                 vision_data: Vision, # Type hint changed
                 context: Dict[str, Any]):
        self.heading_to_hold = context.get('initial_heading', sensors.heading)
        # --- Use Vision methods ---
        if vision_data.is_pole_visible():
            self.last_pole_center_x = vision_data.get_pole_center_x()
        else:
             self.last_pole_center_x = None 
        # ---

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, 
                vision_data: Vision, # Type hint changed
                config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        if self.heading_to_hold is None: self.heading_to_hold = sensors.heading

        # --- Use Vision methods ---
        pole_visible = vision_data.is_pole_visible()
        pole_center_x = vision_data.get_pole_center_x()
        # ---
        cam_w, _ = sensors.camera_image.get_size()

        pole_disappeared_right = (not pole_visible) and \
                                 (self.last_pole_center_x is not None) and \
                                 (self.last_pole_center_x > cam_w * self.disappear_threshold_fraction)

        if pole_disappeared_right:
            return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
        
        commands = sub.get_heading_commands(sensors, self.heading_to_hold, self.surge_power)
        
        if pole_visible and pole_center_x is not None:
            self.last_pole_center_x = pole_center_x
        elif not pole_visible:
             self.last_pole_center_x = None 

        return SubtaskStatus.RUNNING, commands

class SpinUntilTargetReappearsLeft(Subtask):
    """
    Applies constant yaw until the vision target (assumed pole) becomes visible AND
    its center_x is less than a target fraction of the screen width.
    """
    def __init__(self, target_fraction: float, yaw_power: float = -0.3): # Default left yaw
        self.target_fraction = target_fraction
        self.yaw_power = yaw_power

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, 
                vision_data: Vision, # Type hint changed
                config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        
        # --- Use Vision methods ---
        pole_visible = vision_data.is_pole_visible()
        pole_center_x = vision_data.get_pole_center_x()
        # ---
        cam_w, _ = sensors.camera_image.get_size()
        target_pixel_x = cam_w * self.target_fraction

        if pole_visible and pole_center_x is not None and pole_center_x < target_pixel_x:
            return SubtaskStatus.COMPLETED, sub.get_spin_damping_commands(sensors)
            
        commands = sub._mix_and_normalize_commands(0.0, 0.0, self.yaw_power)
        return SubtaskStatus.RUNNING, commands
        
# ApproachAndCenterObject remains conceptually similar but uses Vision methods
class ApproachAndCenterObject(Subtask):
    """
    Surges and steers to center an object (50%) based on its visual HEIGHT.
    Uses a P-controller on surge to slow down as it approaches the target height.
    Completes when height is within tolerance. Assumes pole is the target.
    """
    def __init__(self, 
                 height_threshold_px: int, 
                 surge_p_gain: float = 0.1, 
                 height_tolerance_px: int = 5,
                 yaw_gain: float = 1.5,
                 lost_timeout: float = 2.0):
        self.height_threshold = height_threshold_px
        self.surge_p_gain = surge_p_gain
        self.height_tolerance = height_tolerance_px
        self.yaw_gain = yaw_gain
        self.lost_timeout = lost_timeout
        self.time_since_target_lost = 0.0

    def on_enter(self, sub: 'Submarine', sensors: SensorSuite, 
                 vision_data: Vision, # Type hint changed
                 context: Dict[str, Any]):
        self.time_since_target_lost = 0.0
        context['initial_heading'] = sensors.heading

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, 
                vision_data: Vision, # Type hint changed
                config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        
        # --- Use Vision methods ---
        # Assumes pole is the target for this subtask
        target_height = vision_data.get_pole_apparent_height() 
        target_center_x = vision_data.get_pole_center_x()
        is_visible = vision_data.is_pole_visible()
        # ---

        if is_visible and target_center_x is not None:
            # --- Target is visible ---
            self.time_since_target_lost = 0.0
            
            # --- SURGE P-Controller ---
            height_error = self.height_threshold - target_height
            surge = height_error * self.surge_p_gain
            surge = np.clip(surge, -0.4, 0.5) 
            
            # --- YAW P-Controller ---
            cam_w, _ = sensors.camera_image.get_size()
            pixel_error_x = target_center_x - (cam_w / 2)
            yaw_p = -(pixel_error_x / (cam_w / 2)) * self.yaw_gain
            yaw_d = -sensors.imu.gyro_z * sub.YAW_D_GAIN
            yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)

            # --- Check for completion ---
            is_at_dist = abs(height_error) < self.height_tolerance
            is_centered = abs(pixel_error_x) < 15 
            
            if is_at_dist and is_centered:
                context['initial_heading'] = sensors.heading
                return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
            
            return SubtaskStatus.RUNNING, sub._mix_and_normalize_commands(surge, 0.0, yaw)
            
        else:
            # --- Target is lost ---
            self.time_since_target_lost += dt
            if self.time_since_target_lost > self.lost_timeout:
                print(f"ERROR: ApproachAndCenterObject failed - target lost for > {self.lost_timeout}s")
                return SubtaskStatus.FAILED, sub.get_spin_damping_commands(sensors)
            
            return SubtaskStatus.RUNNING, sub.get_spin_damping_commands(sensors)

    def on_exit(self, sub: 'Submarine', sensors: SensorSuite, 
                vision_data: Vision, # Type hint changed
                context: Dict[str, Any]):
        context['initial_heading'] = sensors.heading