#!/usr/bin/env python3
"""
Contains the "brain" of the sub: the Submarine class that manages the mission plan and AI logic.
"""
import math
import numpy as np
from typing import Tuple, List, Optional

# --- Import Vision class ---
from data_structures import ThrusterCommands, SensorSuite, Vision
# ---
from config import SimulationConfig
from utils import angle_diff
from ai.tasks import TaskStatus # Corrected import
from ai.tasks.task_base import Task

class Submarine:
    def __init__(self, mission_plan: List[Task]):
        # PID and Control Gains 
        self.ALIGN_YAW_P_GAIN = 0.6
        self.ALIGN_SWAY_P_GAIN = 1.8 # Likely unused for surface boat
        self.ALIGN_DAMPING_GAIN = 1.0 # Likely unused for surface boat
        self.YAW_D_GAIN = 4.0 # Keep increased value
        self.HOVER_YAW_P_GAIN = 0.1
        self.HOVER_XY_P_GAIN = 1.0
        self.HOVER_XY_I_GAIN = 0.0
        self.HOVER_XY_D_GAIN = 4.0
        self.DAMPING_GAIN = 1.5
        self.MANEUVER_DAMPING_GAIN = 3.0
        self.MIN_PIXELS_FOR_DETECTION = 20 # Used by Vision class now

        self.mission_plan = mission_plan
        self.config = SimulationConfig()
        
        # --- Store latest sensors ---
        self._latest_sensors: Optional[SensorSuite] = None
        # ---
        
        # --- Create Vision object ---
        # The lambda function captures 'self' to access _latest_sensors later
        self.vision = Vision(image_provider=lambda: self._latest_sensors.camera_image if self._latest_sensors else None,
                             min_pole_pixels=self.MIN_PIXELS_FOR_DETECTION,
                             min_gate_pixels=50) # Example gate threshold
        # ---
        
        self.reset()

    def reset(self):
        self.current_task_index = 0
        for task in self.mission_plan:
            task.reset()
        self.target_x, self.target_y, self.target_heading = 0.0, 0.0, 0.0 
        self.integral_x_err, self.integral_y_err, self.integral_clamp = 0.0, 0.0, 2.0 
        self._latest_sensors = None # Clear sensors on reset

    def update(self, dt: float, sensors: SensorSuite) -> Tuple[ThrusterCommands, Vision]: # Return Vision object for debug
        
        self._latest_sensors = sensors

        if self.current_task_index >= len(self.mission_plan):
            return ThrusterCommands(), self.vision 

        current_task = self.mission_plan[self.current_task_index]

        self.vision.update()

        status, commands = current_task.execute(self, dt, sensors, self.vision, self.config)

        # --- Task Transition Logic ---
        if status == TaskStatus.COMPLETED and self.current_task_index < len(self.mission_plan) - 1:
            
            # --- ADD Logic to set search direction ---
            # Check if the completed task was RatchetTurnTask
            from ai.tasks import RatchetTurnTask, GateTask # Add imports here
            if isinstance(current_task, RatchetTurnTask):
                # Check if the *next* task is GateTask
                next_task_index = self.current_task_index + 1
                if next_task_index < len(self.mission_plan):
                    next_task_instance = self.mission_plan[next_task_index]
            # --- END Logic addition ---

            # Standard transition
            self.current_task_index += 1
            next_task = self.mission_plan[self.current_task_index]
            if hasattr(next_task, 'SPEED_THRESHOLD'): 
                 self.integral_x_err, self.integral_y_err = 0.0, 0.0
            if hasattr(next_task, 'on_start'):
                next_task.on_start(self, sensors) 

        return commands, self.vision
    # (Helper methods like get_current_task_name, get_current_state_name, etc., remain the same)
    # ... [rest of the helper methods] ...
    
    def get_current_task_name(self) -> str:
        if self.current_task_index < len(self.mission_plan):
             return self.mission_plan[self.current_task_index].__class__.__name__
        return "MISSION_COMPLETE"

    def get_current_state_name(self) -> str:
        if self.current_task_index < len(self.mission_plan):
             return self.mission_plan[self.current_task_index].state_name
        return ""

    def _get_navigation_target(self, vision: Vision, cam_w: int) -> Tuple[float | None, str | None]:
        # Example using new Vision object - likely unused now
        gate_pair = vision.get_gate_pair()
        if gate_pair:
             l, r = gate_pair[0]['center_x'], gate_pair[1]['center_x']
             return (l, 'left') if abs(l - cam_w/2) < abs(r - cam_w/2) else (r, 'right')
        return (None, None)

    def get_search_commands(self, sensors, target_heading=None, surge_power=0.0):
        # SEARCH_TURN_COMMAND and SCAN_TURN_COMMAND might need to be added back
        SEARCH_TURN_COMMAND = 0.25 
        SCAN_TURN_COMMAND = 0.2
        yaw = SEARCH_TURN_COMMAND
        if target_heading:
             yaw_error = angle_diff(target_heading, sensors.heading)
             yaw_command = yaw_error * self.HOVER_YAW_P_GAIN
             yaw = np.clip(yaw_command, -SCAN_TURN_COMMAND, SCAN_TURN_COMMAND)
        return self._mix_and_normalize_commands(surge_power, 0, yaw)

    def get_spin_damping_commands(self, sensors: SensorSuite):
        h_rad, cos_h, sin_h = math.radians(sensors.heading), math.cos(math.radians(sensors.heading)), math.sin(math.radians(sensors.heading))
        wd_x, wd_y = -sensors.velocity_x * self.DAMPING_GAIN, -sensors.velocity_y * self.DAMPING_GAIN # Use general damping
        surge, sway = wd_x * cos_h + wd_y * sin_h, -wd_x * sin_h + wd_y * cos_h
        yaw = -sensors.imu.gyro_z * self.YAW_D_GAIN
        return self._mix_and_normalize_commands(surge, sway, yaw)

    def get_heading_commands(self, sensors, heading, surge_power=0.0):
        # --- ADD THESE LINES ---
        h_rad = math.radians(sensors.heading)
        cos_h, sin_h = math.cos(h_rad), math.sin(h_rad)
        # --- END ADD ---

        yaw_err = angle_diff(heading, sensors.heading)
        yaw_p = yaw_err * self.HOVER_YAW_P_GAIN
        yaw_d = -sensors.imu.gyro_z * self.YAW_D_GAIN # Add damping
        yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)
        
        # Add sway damping if needed for straight driving
        sway_vel = -sensors.velocity_x * sin_h + sensors.velocity_y * cos_h
        sway_damp = -sway_vel * self.MANEUVER_DAMPING_GAIN # Use maneuver damping for sideways motion
        sway = np.clip(sway_damp, -0.5, 0.5) # Limit sway damping effect
        
        # --- Rename local variable ---
        vision = self.vision # Example, although not used in this specific method
        # ---
        
        return self._mix_and_normalize_commands(surge_power, sway, yaw)
    def _get_pid_hover_commands(self, sensors: SensorSuite, dt: float, tx: float, ty: float,
                                yaw_p_gain_override: Optional[float] = None) -> ThrusterCommands:
        yaw_p_gain = yaw_p_gain_override if yaw_p_gain_override is not None else self.HOVER_YAW_P_GAIN
        ex, ey = tx - sensors.x, ty - sensors.y
        self.integral_x_err = np.clip(self.integral_x_err + ex * dt, -self.integral_clamp, self.integral_clamp)
        self.integral_y_err = np.clip(self.integral_y_err + ey * dt, -self.integral_clamp, self.integral_clamp)
        wtx = (ex * self.HOVER_XY_P_GAIN) + (self.integral_x_err * self.HOVER_XY_I_GAIN) - (sensors.velocity_x * self.HOVER_XY_D_GAIN)
        wty = (ey * self.HOVER_XY_P_GAIN) + (self.integral_y_err * self.HOVER_XY_I_GAIN) - (sensors.velocity_y * self.HOVER_XY_D_GAIN)
        h_rad, c, s = math.radians(sensors.heading), math.cos(math.radians(sensors.heading)), math.sin(math.radians(sensors.heading))
        fsh, sway = wtx * c + wty * s, -wtx * s + wty * c
        # Yaw control for hover
        yaw_err = angle_diff(self.target_heading, sensors.heading)
        yaw_p = yaw_err * yaw_p_gain
        yaw_d = -sensors.imu.gyro_z * self.YAW_D_GAIN
        yaw_cmd = np.clip(yaw_p + yaw_d, -1.0, 1.0)
        
        surge = fsh
        return self._mix_and_normalize_commands(surge, sway, yaw_cmd)

    def _get_damping_commands(self, sensors: SensorSuite) -> ThrusterCommands:
        # Simple velocity damping
        wdx, wdy = -sensors.velocity_x * self.DAMPING_GAIN, -sensors.velocity_y * self.DAMPING_GAIN
        h_rad = math.radians(sensors.heading)
        c, s = math.cos(h_rad), math.sin(h_rad)
        fsh, sway = wdx*c+wdy*s, -wdx*s+wdy*c
        # Yaw damping only (no P-gain to hold heading unless explicitly requested)
        yaw = -sensors.imu.gyro_z * self.YAW_D_GAIN 
        surge = fsh
        return self._mix_and_normalize_commands(surge, sway, yaw)

    def _mix_and_normalize_commands(self, surge, sway, yaw) -> ThrusterCommands:
        # Simple differential drive mixing (ignoring sway command for now)
        # TODO: Implement sway mixing if 4 thrusters were used
        commands = ThrusterCommands()
        commands.port = surge + yaw
        commands.starboard = surge - yaw
        # Normalize
        max_abs = max(1.0, abs(commands.port), abs(commands.starboard))
        if max_abs > 1.0:
             commands.port /= max_abs
             commands.starboard /= max_abs
        return commands

    def get_go_to_visual_target_commands(self, sensors: SensorSuite, nav_target_x: float, surge_power: float):
        cam_w, cam_h = sensors.camera_image.get_size()
        pixel_error_x = nav_target_x - (cam_w / 2)
        # Yaw P controller
        yaw_p = -(pixel_error_x / (cam_w / 2)) * self.ALIGN_YAW_P_GAIN # Use align gain
        # Yaw D controller
        yaw_d = -sensors.imu.gyro_z * self.YAW_D_GAIN
        yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)
        # Sway Damping (resist sideways motion)
        h_rad = math.radians(sensors.heading)
        sway_vel = -sensors.velocity_x * math.sin(h_rad) + sensors.velocity_y * math.cos(h_rad)
        sway = -sway_vel * self.MANEUVER_DAMPING_GAIN # Use maneuver damping
        sway = np.clip(sway, -0.5, 0.5) # Limit damping effect
        
        return self._mix_and_normalize_commands(surge_power, sway, yaw)