#!/usr/bin/env python3
"""
Implementations of common, reusable Subtasks, using context.
Subtasks now expect two Vision objects.
"""
import math
from typing import Tuple, List, Optional, Dict, Any
import numpy as np

# Absolute import for the base class
from ai.tasks.subtask_base import Subtask, SubtaskStatus
# Import Vision class
from data_structures import SensorSuite, Vision, ThrusterCommands
#
from config import SimulationConfig
from utils import angle_diff

# --- Navigation Subtasks ---
class TurnToHeading(Subtask):
    def __init__(self, absolute_degrees: Optional[float] = None, relative_degrees: Optional[float] = None, tolerance_degrees: float = 5.0):
        if absolute_degrees is not None and relative_degrees is not None: raise ValueError("Provide either absolute_degrees or relative_degrees, not both.")
        if absolute_degrees is None and relative_degrees is None: raise ValueError("Must provide either absolute_degrees or relative_degrees.")
        self.absolute_degrees=absolute_degrees; self.relative_degrees=relative_degrees; self.tolerance=tolerance_degrees; self.target_heading: Optional[float] = None
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite, front_vision: Vision, side_vision: Vision, context: Dict[str, Any]):
        if self.absolute_degrees is not None: self.target_heading = self.absolute_degrees % 360
        elif self.relative_degrees is not None:
            initial_heading = context.get('initial_heading');
            if initial_heading is None: initial_heading = sensors.heading
            self.target_heading = (initial_heading + self.relative_degrees) % 360
        else: self.target_heading = None
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, front_vision: Vision, side_vision: Vision, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        if self.target_heading is None: print("ERROR: TurnToHeading target could not be determined!"); return SubtaskStatus.FAILED, sub._get_damping_commands(sensors)
        heading_error = angle_diff(self.target_heading, sensors.heading)
        if abs(heading_error) < self.tolerance: return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
        commands = sub.get_heading_commands(sensors, self.target_heading, surge_power=0.0)
        return SubtaskStatus.RUNNING, commands
    def get_dynamic_name(self, context: Dict[str, Any]) -> str:
        if self.target_heading is not None: return f"{self.name}({self.target_heading:.0f}°)"
        elif self.absolute_degrees is not None: return f"{self.name}(Abs {self.absolute_degrees:.0f}°)"
        elif self.relative_degrees is not None:
            rel = self.relative_degrees
            init_h = context.get('initial_heading')
            if init_h is None:
                return f"{self.name}(Rel {rel:.0f}° from ?)"
            calc_target = f" -> {(init_h + rel) % 360:.0f}"
            return f"{self.name}(Rel {rel:.0f}° from {init_h:.0f}°{calc_target})"
        return self.name

class DriveStraight(Subtask):
    def __init__(self, duration: float, surge_power: float = 0.5):
        self.duration=duration; self.surge_power=surge_power; self.timer=0.0; self.heading_to_hold=None
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite, front_vision: Vision, side_vision: Vision, context: Dict[str, Any]):
        self.timer = self.duration; self.heading_to_hold = sensors.heading
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, front_vision: Vision, side_vision: Vision, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        if self.timer <= 0: return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
        self.timer -= dt
        if self.heading_to_hold is None: self.heading_to_hold = sensors.heading
        commands = sub.get_heading_commands(sensors, self.heading_to_hold, self.surge_power)
        return SubtaskStatus.RUNNING, commands

class Stabilize(Subtask):
    def __init__(self, duration: float = 2.0, speed_threshold: float = 0.05):
        self.duration=duration; self.speed_threshold=speed_threshold; self.timer=0.0; self.target_set=False
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite, front_vision: Vision, side_vision: Vision, context: Dict[str, Any]):
        self.timer=0.0; self.target_set=False; sub.integral_x_err, sub.integral_y_err = 0.0, 0.0
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, front_vision: Vision, side_vision: Vision, config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        if not self.target_set: sub.target_x, sub.target_y = sensors.x, sensors.y; sub.target_heading = sensors.heading; self.target_set = True
        self.timer += dt; speed = math.hypot(sensors.velocity_x, sensors.velocity_y)
        if self.timer > self.duration and speed < self.speed_threshold: return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
        commands = sub._get_pid_hover_commands(sensors, dt, sub.target_x, sub.target_y)
        return SubtaskStatus.RUNNING, commands

# --- Vision-Based Subtasks ---
class WaitForTargetVisible(Subtask):
    def __init__(self, target_type='either'):
        self.target_type = target_type.lower()
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite,
                front_vision: Vision, side_vision: Vision,
                config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        visible = False
        if self.target_type == 'pole':
            visible = front_vision.is_pole_visible()
        elif self.target_type == 'gate':
            visible = front_vision.is_gate_visible()
        else: # 'either'
            visible = front_vision.is_pole_visible() or front_vision.is_gate_visible()
        if visible:
            return SubtaskStatus.COMPLETED, sub.get_spin_damping_commands(sensors)
        else:
            return SubtaskStatus.RUNNING, ThrusterCommands()

class AlignToObjectX(Subtask):
    def __init__(self, target_x_fraction: float, tolerance_px: int = 15, yaw_gain: float = 0.8, yaw_rate_tolerance: float = 0.05):
        self.target_x_fraction=target_x_fraction; self.tolerance_px=tolerance_px; self.yaw_gain=yaw_gain; self.yaw_rate_tolerance=yaw_rate_tolerance
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite,
                front_vision: Vision, side_vision: Vision,
                config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        current_center_x = None
        if front_vision.is_gate_visible():
            current_center_x = front_vision.get_gate_center_x()
        elif front_vision.is_pole_visible():
            current_center_x = front_vision.get_pole_center_x()
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
                front_vision: Vision, side_vision: Vision,
                context: Dict[str, Any]):
        context['initial_heading'] = sensors.heading

class DriveUntilTargetLost(Subtask): # Used by older GateTask, potentially still useful
    def __init__(self, surge_power: float = 0.5, target_type: str = 'gate'):
        if target_type not in ['gate', 'pole', 'either']:
             raise ValueError("target_type must be 'gate', 'pole', or 'either'")
        self.surge_power = surge_power
        self.target_type = target_type
        self.heading_to_hold: Optional[float] = None
        self.target_was_visible = False
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite,
                 front_vision: Vision, side_vision: Vision,
                 context: Dict[str, Any]):
        self.heading_to_hold = context.get('initial_heading', sensors.heading)
        if self.target_type == 'gate':
             self.target_was_visible = front_vision.is_gate_visible()
        elif self.target_type == 'pole':
             self.target_was_visible = front_vision.is_pole_visible()
        else: # either
             self.target_was_visible = front_vision.is_gate_visible() or front_vision.is_pole_visible()
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite,
                front_vision: Vision, side_vision: Vision,
                config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        if self.heading_to_hold is None: self.heading_to_hold = sensors.heading
        currently_visible = False
        if self.target_type == 'gate':
             currently_visible = front_vision.is_gate_visible()
        elif self.target_type == 'pole':
             currently_visible = front_vision.is_pole_visible()
        else: # either
             currently_visible = front_vision.is_gate_visible() or front_vision.is_pole_visible()
        if currently_visible:
            self.target_was_visible = True
            commands = sub.get_heading_commands(sensors, self.heading_to_hold, self.surge_power)
            return SubtaskStatus.RUNNING, commands
        else:
            if self.target_was_visible:
                return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
            else:
                print(f"ERROR: DriveUntilTargetLost failed - target '{self.target_type}' never visible.")
                return SubtaskStatus.FAILED, sub._get_damping_commands(sensors)

class DriveUntilTargetLostRight(Subtask): # Used by RatchetTurn (currently unused)
    def __init__(self, surge_power: float = 0.3, disappear_threshold_fraction: float = 0.95):
        self.surge_power = surge_power
        self.disappear_threshold_fraction = disappear_threshold_fraction
        self.heading_to_hold: Optional[float] = None
        self.last_pole_center_x: Optional[float] = None
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite,
                 front_vision: Vision, side_vision: Vision,
                 context: Dict[str, Any]):
        self.heading_to_hold = context.get('initial_heading', sensors.heading)
        if front_vision.is_pole_visible():
            self.last_pole_center_x = front_vision.get_pole_center_x()
        else:
             self.last_pole_center_x = None
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite,
                front_vision: Vision, side_vision: Vision,
                config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        if self.heading_to_hold is None: self.heading_to_hold = sensors.heading
        pole_visible = front_vision.is_pole_visible()
        pole_center_x = front_vision.get_pole_center_x()
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

class SpinUntilTargetReappearsLeft(Subtask): # Used by RatchetTurn (currently unused)
    def __init__(self, target_fraction: float, yaw_power: float = -0.3): # Default left yaw
        self.target_fraction = target_fraction
        self.yaw_power = yaw_power
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite,
                front_vision: Vision, side_vision: Vision,
                config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        pole_visible = front_vision.is_pole_visible()
        pole_center_x = front_vision.get_pole_center_x()
        cam_w, _ = sensors.camera_image.get_size()
        target_pixel_x = cam_w * self.target_fraction
        if pole_visible and pole_center_x is not None and pole_center_x < target_pixel_x:
            return SubtaskStatus.COMPLETED, sub.get_spin_damping_commands(sensors)
        commands = sub._mix_and_normalize_commands(0.0, 0.0, self.yaw_power)
        return SubtaskStatus.RUNNING, commands

class ApproachObjectByWidth(Subtask):
    def __init__(self,
                 target_width_px: int,
                 surge_p_gain: float = 0.1,
                 width_tolerance_px: int = 5,
                 yaw_gain: float = 1.5,
                 lost_timeout: float = 2.0):
        self.target_width_px = target_width_px
        self.surge_p_gain = surge_p_gain
        self.width_tolerance_px = width_tolerance_px
        self.yaw_gain = yaw_gain
        self.lost_timeout = lost_timeout
        self.time_since_target_lost = 0.0
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite,
                 front_vision: Vision, side_vision: Vision,
                 context: Dict[str, Any]):
        self.time_since_target_lost = 0.0
        context['initial_heading'] = sensors.heading
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite,
                front_vision: Vision, side_vision: Vision,
                config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        pole = front_vision.get_best_pole()
        is_visible = pole is not None
        if is_visible:
            target_width = pole['width']
            target_center_x = pole['center_x']
            self.time_since_target_lost = 0.0
            width_error = self.target_width_px - target_width
            surge = width_error * self.surge_p_gain
            surge = np.clip(surge, -0.4, 0.5)
            cam_w, _ = sensors.camera_image.get_size()
            pixel_error_x = target_center_x - (cam_w / 2)
            yaw_p = -(pixel_error_x / (cam_w / 2)) * self.yaw_gain
            yaw_d = -sensors.imu.gyro_z * sub.YAW_D_GAIN
            yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)
            is_at_dist = abs(width_error) < self.width_tolerance_px
            is_centered = abs(pixel_error_x) < 15
            if is_at_dist and is_centered:
                context['initial_heading'] = sensors.heading
                return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
            return SubtaskStatus.RUNNING, sub._mix_and_normalize_commands(surge, 0.0, yaw)
        else:
            self.time_since_target_lost += dt
            if self.time_since_target_lost > self.lost_timeout:
                print(f"ERROR: ApproachObjectByWidth failed - target lost for > {self.lost_timeout}s")
                return SubtaskStatus.FAILED, sub._get_damping_commands(sensors)
            # Continue searching using spin from Task base class
            return SubtaskStatus.RUNNING, ThrusterCommands()
    def on_exit(self, sub: 'Submarine', sensors: SensorSuite,
                front_vision: Vision, side_vision: Vision,
                context: Dict[str, Any]):
        context['initial_heading'] = sensors.heading

# --- Side Camera Subtasks ---
class AlignUsingSideCamera(Subtask):
    def __init__(self,
                 target_x_fraction: float,
                 tolerance_px: int = 15,
                 search_yaw_power: float = -0.3,
                 align_yaw_gain: float = 1.8,
                 yaw_rate_tolerance: float = 0.05):
        self.target_x_fraction = target_x_fraction
        self.tolerance_px = tolerance_px
        self.search_yaw_power = search_yaw_power
        self.align_yaw_gain = align_yaw_gain
        self.yaw_rate_tolerance = yaw_rate_tolerance
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite,
                 front_vision: Vision, side_vision: Vision,
                 context: Dict[str, Any]):
        pass # Simplified
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite,
                front_vision: Vision, side_vision: Vision,
                config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        if sensors.side_camera_image is None:
             return SubtaskStatus.FAILED, sub._get_damping_commands(sensors)
        pole_visible_side = side_vision.is_pole_visible_side()

        if pole_visible_side:
            pole_x = side_vision.get_pole_center_x_side()
            if pole_x is not None:
                cam_w, _ = sensors.side_camera_image.get_size()
                target_pixel_x = cam_w * self.target_x_fraction
                pixel_error_x = pole_x - target_pixel_x
                is_centered = abs(pixel_error_x) < self.tolerance_px
                is_stable = abs(sensors.imu.gyro_z) < self.yaw_rate_tolerance

                if is_centered and is_stable:
                    return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
                yaw_p = -(pixel_error_x / (cam_w / 2)) * self.align_yaw_gain
                yaw_d = -sensors.imu.gyro_z * sub.YAW_D_GAIN
                yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)
                return SubtaskStatus.RUNNING, sub._mix_and_normalize_commands(0.0, 0.0, yaw)
            else:
                 # Pole visible but center_x is None? Search.
                 return SubtaskStatus.RUNNING, sub._mix_and_normalize_commands(0.0, 0.0, self.search_yaw_power)
        # --- Target not visible ---
        return SubtaskStatus.RUNNING, sub._mix_and_normalize_commands(0.0, 0.0, self.search_yaw_power)
    def on_exit(self, sub: 'Submarine', sensors: SensorSuite,
                front_vision: Vision, side_vision: Vision,
                context: Dict[str, Any]):
        context['initial_heading'] = sensors.heading

class OrbitUsingSideCamera(Subtask):
    def __init__(self,
                 target_pole_width_px: int,
                 orbit_surge_power: float,
                 yaw_x_gain: float,         # P-gain for centering (should be positive)
                 yaw_dist_p_gain: float,    # P-gain for distance (should be negative)
                 yaw_dist_i_gain: float,    # I-gain for distance (should be negative)
                 local_search_yaw_power: float = -0.2, # Yaw left if pole lost during orbit
                 lost_timeout: float = 3.0,
                 integral_clamp: float = 2.0): # Keep increased clamp
        self.target_pole_width_px = target_pole_width_px
        self.orbit_surge_power = orbit_surge_power
        self.yaw_x_gain = yaw_x_gain
        self.yaw_dist_p_gain = yaw_dist_p_gain
        self.yaw_dist_i_gain = yaw_dist_i_gain
        self.local_search_yaw_power = local_search_yaw_power
        self.lost_timeout = lost_timeout
        self.integral_clamp = integral_clamp
        self.time_since_target_lost = 0.0
        self.integral_dist_err = 0.0
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite,
                 front_vision: Vision, side_vision: Vision,
                 context: Dict[str, Any]):
        self.time_since_target_lost = 0.0
        self.integral_dist_err = 0.0
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite,
                front_vision: Vision, side_vision: Vision,
                config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        if front_vision.is_gate_visible():
            return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
        if not side_vision.is_pole_visible_side():
            self.time_since_target_lost += dt
            if self.time_since_target_lost > self.lost_timeout:
                print(f"ERROR: OrbitUsingSideCamera lost target for {self.lost_timeout}s.")
                return SubtaskStatus.FAILED, sub.get_spin_damping_commands(sensors)
            return SubtaskStatus.RUNNING, sub._mix_and_normalize_commands(0.0, 0.0, self.local_search_yaw_power)
        if self.time_since_target_lost > 0:
            self.integral_dist_err = 0.0
        self.time_since_target_lost = 0.0
        pole = side_vision.get_best_pole_side()
        if pole is None:
            return SubtaskStatus.RUNNING, sub._mix_and_normalize_commands(0.0, 0.0, self.local_search_yaw_power)
        pole_x = pole['center_x']
        pole_width = pole['width']
        if sensors.side_camera_image is None:
            # Removed redundant print
            return SubtaskStatus.FAILED, sub._get_damping_commands(sensors)
        cam_w, _ = sensors.side_camera_image.get_size()
        target_x = cam_w / 2.0
        error_x = target_x - pole_x
        yaw_p_x = error_x * self.yaw_x_gain
        error_dist = self.target_pole_width_px - pole_width
        self.integral_dist_err += error_dist * dt
        self.integral_dist_err = np.clip(self.integral_dist_err, -self.integral_clamp, self.integral_clamp)
        yaw_p_dist = error_dist * self.yaw_dist_p_gain
        yaw_i_dist = self.integral_dist_err * self.yaw_dist_i_gain
        yaw_d = -sensors.imu.gyro_z * sub.YAW_D_GAIN
        yaw = yaw_p_x + yaw_p_dist + yaw_i_dist + yaw_d
        yaw = np.clip(yaw, -1.0, 1.0)
        surge = self.orbit_surge_power
        return SubtaskStatus.RUNNING, sub._mix_and_normalize_commands(surge, 0.0, yaw)
    def get_dynamic_name(self, context: Dict[str, Any]) -> str:
        search_indicator = " Searching!" if self.time_since_target_lost > 0 else ""
        return f"SideOrbit(FindGate I={self.integral_dist_err:.2f}){search_indicator}"

# --- NEW Gate Driving Subtask ---
class DriveThroughGate(Subtask):
    def __init__(self,
                 surge_power: float = 0.6,
                 yaw_gain: float = 1.5,
                 lost_timeout: float = 1.0):
        self.surge_power = surge_power
        self.yaw_gain = yaw_gain
        self.lost_timeout = lost_timeout
        self.gate_was_visible = False
        self.time_since_last_seen = 0.0
    def on_enter(self, sub: 'Submarine', sensors: SensorSuite,
                 front_vision: Vision, side_vision: Vision,
                 context: Dict[str, Any]):
        self.gate_was_visible = front_vision.is_gate_visible()
        self.time_since_last_seen = 0.0
        context['initial_heading'] = sensors.heading
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite,
                front_vision: Vision, side_vision: Vision,
                config: SimulationConfig, context: Dict[str, Any]) -> Tuple[SubtaskStatus, ThrusterCommands]:
        gate_visible = front_vision.is_gate_visible()
        gate_center_x = front_vision.get_gate_center_x()
        if gate_visible and gate_center_x is not None:
            self.gate_was_visible = True
            self.time_since_last_seen = 0.0
            cam_w, _ = sensors.camera_image.get_size()
            target_x = cam_w / 2.0
            error_x = gate_center_x - target_x
            yaw_p = -(error_x / (cam_w / 2)) * self.yaw_gain
            yaw_d = -sensors.imu.gyro_z * sub.YAW_D_GAIN
            yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)
            commands = sub._mix_and_normalize_commands(self.surge_power, 0.0, yaw)
            return SubtaskStatus.RUNNING, commands
        else:
            self.time_since_last_seen += dt
            if self.gate_was_visible:
                return SubtaskStatus.COMPLETED, sub._get_damping_commands(sensors)
            if self.time_since_last_seen > self.lost_timeout:
                print(f"ERROR: DriveThroughGate failed - gate never visible for {self.lost_timeout}s.")
                return SubtaskStatus.FAILED, sub._get_damping_commands(sensors)
            initial_heading = context.get('initial_heading', sensors.heading)
            commands = sub.get_heading_commands(sensors, initial_heading, self.surge_power)
            return SubtaskStatus.RUNNING, commands
    def on_exit(self, sub: 'Submarine', sensors: SensorSuite,
                 front_vision: Vision, side_vision: Vision,
                 context: Dict[str, Any]):
        context['initial_heading'] = sensors.heading