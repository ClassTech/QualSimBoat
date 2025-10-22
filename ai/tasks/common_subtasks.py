#!/usr/bin/env python3
"""
Implementations of common, reusable Subtasks, using context.
"""
import math
from typing import Tuple, List, Optional, Dict, Any
import numpy as np

# Absolute import for the base class
from ai.tasks.subtask_base import Subtask, SubtaskStatus
# Other necessary imports
from data_structures import SensorSuite, VisionData, ThrusterCommands
from config import SimulationConfig
from utils import angle_diff
# Vision import needed for specific subtasks if they do direct detection
from ai.vision import find_blobs_hsv

# --- Navigation Subtasks ---
# (TurnToHeading, DriveStraight, Stabilize remain the same)
class TurnToHeading(Subtask): # ... as before ...
    def __init__(self, absolute_degrees: Optional[float] = None, relative_degrees: Optional[float] = None, tolerance_degrees: float = 5.0): #...
        if absolute_degrees is not None and relative_degrees is not None: raise ValueError("Provide either absolute_degrees or relative_degrees, not both.")
        if absolute_degrees is None and relative_degrees is None: raise ValueError("Must provide either absolute_degrees or relative_degrees.")
        self.absolute_degrees=absolute_degrees; self.relative_degrees=relative_degrees; self.tolerance=tolerance_degrees; self.target_heading: Optional[float] = None
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite, vision_data: VisionData, context: Dict[str, Any]): #...
        if self.absolute_degrees is not None: self.target_heading = self.absolute_degrees % 360
        elif self.relative_degrees is not None:
            initial_heading = context.get('initial_heading');
            if initial_heading is None: self.target_heading = None
            else: self.target_heading = (initial_heading + self.relative_degrees) % 360
        else: self.target_heading = None
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]: #...
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
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite, vision_data: VisionData, context: Dict[str, Any]): #...
        self.timer = self.duration; self.heading_to_hold = sensors.heading
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]: #...
        if self.timer <= 0: return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
        self.timer -= dt
        if self.heading_to_hold is None: self.heading_to_hold = sensors.heading
        commands = sub.get_heading_commands(sensors, self.heading_to_hold, self.surge_power)
        return SubtaskStatus.RUNNING, commands

class Stabilize(Subtask): # ... as before ...
    def __init__(self, duration: float = 2.0, speed_threshold: float = 0.05): #...
        self.duration=duration; self.speed_threshold=speed_threshold; self.timer=0.0; self.target_set=False
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite, vision_data: VisionData, context: Dict[str, Any]): #...
        self.timer=0.0; self.target_set=False; sub.integral_x_err, sub.integral_y_err = 0.0, 0.0
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]: #...
        if not self.target_set: sub.target_x, sub.target_y = sensors.x, sensors.y; sub.target_heading = sensors.heading; self.target_set = True
        self.timer += dt; speed = math.hypot(sensors.velocity_x, sensors.velocity_y)
        if self.timer > self.duration and speed < self.speed_threshold: return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
        commands = sub._get_pid_hover_commands(sensors, dt, sub.target_x, sub.target_y)
        return SubtaskStatus.RUNNING, commands

# --- Vision-Based Subtasks ---

class WaitForTargetVisible(Subtask): # ... as before ...
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]: #...
        if vision_data.gate_is_visible: return SubtaskStatus.COMPLETED, sub.get_spin_damping_commands(sensors)
        else: return SubtaskStatus.RUNNING, ThrusterCommands()

class AlignToObjectX(Subtask):
    """
    Pivots to center target at target_x_fraction.
    Stores heading in context['initial_heading'] on completion.
    """
    def __init__(self, target_x_fraction: float, tolerance_px: int = 15, yaw_gain: float = 0.8, yaw_rate_tolerance: float = 0.05):
        self.target_x_fraction=target_x_fraction; self.tolerance_px=tolerance_px; self.yaw_gain=yaw_gain; self.yaw_rate_tolerance=yaw_rate_tolerance

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        if not hasattr(vision_data, 'align_target_center_x') or vision_data.align_target_center_x is None:
             # print("WARN: AlignToObjectX running but align_target_center_x not set.") # Debug
             return SubtaskStatus.FAILED, sub.get_spin_damping_commands(sensors)

        cam_w, _ = sensors.camera_image.get_size()
        target_pixel_x = cam_w * self.target_x_fraction
        current_center_x = vision_data.align_target_center_x

        pixel_error_x = current_center_x - target_pixel_x
        yaw_p = -(pixel_error_x / (cam_w / 2)) * self.yaw_gain
        yaw_d = -sensors.imu.gyro_z * sub.YAW_D_GAIN
        yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)

        is_centered = abs(pixel_error_x) < self.tolerance_px
        is_stable = abs(sensors.imu.gyro_z) < self.yaw_rate_tolerance

        if is_centered and is_stable:
             return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)

        return SubtaskStatus.RUNNING, sub._mix_and_normalize_commands(0.0, 0.0, yaw) # Pivot command

    def on_exit(self, sub: 'Submarine', sensors: SensorSuite, vision_data: VisionData, context: Dict[str, Any]):
        context['initial_heading'] = sensors.heading


class ApproachTargetVisualWidth(Subtask): # ... as before, robust version ...
    def __init__(self, width_threshold_px: int, surge_power: float = 0.5, lost_timeout: float = 1.0):
        self.width_threshold=width_threshold_px; self.surge_power=surge_power; self.lost_timeout=lost_timeout
        self.target_heading=None; self.time_since_target_lost=0.0
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite, vision_data: VisionData, context: Dict[str, Any]):
        self.target_heading = context.get('initial_heading', sensors.heading); self.time_since_target_lost = 0.0
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        target_width = getattr(vision_data, 'align_target_width', None)
        if target_width is not None:
            self.time_since_target_lost = 0.0
            if target_width > self.width_threshold: return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
        else:
            self.time_since_target_lost += dt
            if self.time_since_target_lost > self.lost_timeout: print(f"ERROR: ApproachTargetVisualWidth failed - target lost for > {self.lost_timeout}s"); return SubtaskStatus.FAILED, sub.get_spin_damping_commands(sensors)
        if self.target_heading is None: self.target_heading = sensors.heading
        commands = sub.get_heading_commands(sensors, self.target_heading, self.surge_power)
        return SubtaskStatus.RUNNING, commands

class DriveUntilTargetLost(Subtask): # ... as before ...
    def __init__(self, surge_power: float = 0.5):
        self.surge_power = surge_power; self.heading_to_hold = None; self.target_was_visible = False
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite, vision_data: VisionData, context: Dict[str, Any]):
        self.heading_to_hold = context.get('initial_heading', sensors.heading); self.target_was_visible = vision_data.gate_is_visible
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        if self.heading_to_hold is None: self.heading_to_hold = sensors.heading
        if vision_data.gate_is_visible:
            self.target_was_visible = True
            commands = sub.get_heading_commands(sensors, self.heading_to_hold, self.surge_power)
            return SubtaskStatus.RUNNING, commands
        else:
            if self.target_was_visible: return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
            else: print("ERROR: DriveUntilTargetLost started but target not initially visible."); return SubtaskStatus.FAILED, sub._get_damping_commands(sensors)

# --- MODIFIED SUBTASK: Uses HEIGHT for distance control ---
class ApproachAndCenterObject(Subtask):
    """
    Surges and steers to center an object (50%) based on its visual HEIGHT.
    Uses a P-controller on surge to slow down as it approaches the target height.
    Completes when height is within tolerance.
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

    def on_enter(self, sub: 'Submarine', sensors: SensorSuite, vision_data: VisionData, context: Dict[str, Any]):
        self.time_since_target_lost = 0.0
        context['initial_heading'] = sensors.heading

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        # --- MODIFICATION: Use 'apparent_height' not 'align_target_width' ---
        target_height = getattr(vision_data, 'apparent_height', None)
        target_center_x = getattr(vision_data, 'align_target_center_x', None)

        if target_height is not None and target_center_x is not None:
            # --- Target is visible ---
            self.time_since_target_lost = 0.0
            
            # --- SURGE P-Controller (slows as it approaches) ---
            height_error = self.height_threshold - target_height
            surge = height_error * self.surge_p_gain
            surge = np.clip(surge, -0.4, 0.5) # Max 0.5 forward, 0.4 reverse
            
            # --- YAW P-Controller (steers to center) ---
            cam_w, _ = sensors.camera_image.get_size()
            pixel_error_x = target_center_x - (cam_w / 2)
            yaw_p = -(pixel_error_x / (cam_w / 2)) * self.yaw_gain
            yaw_d = -sensors.imu.gyro_z * sub.YAW_D_GAIN
            yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)

            # --- Check for completion ---
            is_at_dist = abs(height_error) < self.height_tolerance
            is_centered = abs(pixel_error_x) < 15 # 15px tolerance for center
            
            if is_at_dist and is_centered:
                context['initial_heading'] = sensors.heading
                return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
            
            # Not complete, so continue approaching
            return SubtaskStatus.RUNNING, sub._mix_and_normalize_commands(surge, 0.0, yaw)
            
        else:
            # --- Target is lost ---
            self.time_since_target_lost += dt
            if self.time_since_target_lost > self.lost_timeout:
                print(f"ERROR: ApproachAndCenterObject failed - target lost for > {self.lost_timeout}s")
                return SubtaskStatus.FAILED, sub.get_spin_damping_commands(sensors)
            
            # Target lost, just damp spin and wait to reacquire
            return SubtaskStatus.RUNNING, sub.get_spin_damping_commands(sensors)

    def on_exit(self, sub: 'Submarine', sensors: SensorSuite, vision_data: VisionData, context: Dict[str, Any]):
        # Store the final heading for the next task
        context['initial_heading'] = sensors.heading