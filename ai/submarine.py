#!/usr/bin/env python3
"""
Contains the "brain" of the sub: the Submarine class that manages the mission plan and AI logic.
"""
import math
import numpy as np
from typing import Tuple, List, Optional

from data_structures import ThrusterCommands, SensorSuite, VisionData
from config import SimulationConfig
from utils import angle_diff
from ai.tasks import TaskStatus # Corrected import
from ai.tasks.task_base import Task



class Submarine:
    def __init__(self, mission_plan: List[Task]):
        # PID and Control Gains (Keep the tuned values)
        self.ALIGN_YAW_P_GAIN = 0.6
        self.ALIGN_SWAY_P_GAIN = 1.8
        self.ALIGN_DAMPING_GAIN = 1.0
        self.YAW_D_GAIN = 4.5
        self.HOVER_YAW_P_GAIN = 0.1
        self.HOVER_XY_P_GAIN = 1.0
        self.HOVER_XY_I_GAIN = 0.0
        self.HOVER_XY_D_GAIN = 4.0
        self.DAMPING_GAIN = 1.5
        self.MANEUVER_DAMPING_GAIN = 3.0
        self.SURGE_MIN_SPEED = 0.3
        self.SURGE_MAX_SPEED = 0.8
        self.SEARCH_TURN_COMMAND = 0.25
        self.SCAN_TURN_COMMAND = 0.2
        self.SLALOM_NAV_YAW_GAIN = 1.5 # Unused now?
        self.SLALOM_SWAY_D_GAIN = 2.0  # Unused now?
        self.ALIGN_POLE_HEIGHT_P_GAIN = 0.02 # Unused now?
        self.ALIGN_HEADING_TOLERANCE = 3.0 # Unused now?
        self.ALIGN_POS_TOLERANCE = 0.08 # Unused now?
        self.ALIGN_SPEED_TOLERANCE_SQ = 0.0025 # Unused now?
        self.MIN_PIXELS_FOR_DETECTION = 20 # Keep for vision subtasks

        self.mission_plan = mission_plan
        self.config = SimulationConfig()
        self.reset()

    def reset(self):
        self.current_task_index = 0
        for task in self.mission_plan:
            task.reset()
        self.gateCompleted = False # Keep if GateTask uses it?
        # self.lastApparentSize = 0.0 # Remove if not used
        # self.last_error_x = 0.0 # Remove if not used
        # self.dance_center_heading = 0.0 # Remove if not used
        # self.gate_passage_side = 'left' # Remove if not used
        self.target_x, self.target_y, self.target_heading = 0.0, 0.0, 0.0 # Keep for Stabilize/Damping
        self.integral_x_err, self.integral_y_err, self.integral_clamp = 0.0, 0.0, 2.0 # Keep for Stabilize PID
        self.approach_heading = 0.0 # Keep if GateTask uses it?
        # self.pass_start_pos, self.pass_end_pos = None, None # Remove if not used
        # self.course_heading = 0.0 # Remove if SlalomTask removed
        # self.slalom_pass_side = None # Remove if SlalomTask removed

    def update(self, dt: float, sensors: SensorSuite) -> Tuple[ThrusterCommands, VisionData]:
        if self.current_task_index >= len(self.mission_plan):
            # Return empty vision data when mission complete
            return ThrusterCommands(), VisionData()

        current_task = self.mission_plan[self.current_task_index]

        # --- MODIFICATION: REMOVED process_vision call ---
        # The Task.execute method now handles calling process_vision internally.
        # ---

        # Pass dummy VisionData; Task.execute will ignore it and use processed data
        status, commands = current_task.execute(self, dt, sensors, VisionData(), self.config)

        # --- MODIFICATION: Task.execute now returns processed vision data ---
        # For debug drawing, we need the vision data the task *actually* used.
        # Get it from the task instance after execution.
        processed_vision_data = VisionData() # Default empty
        if hasattr(current_task, 'gate_vision_data'): # Example for GateTask
            processed_vision_data = current_task.gate_vision_data
        elif hasattr(current_task, 'marker_vision_data'): # Example for MarkerTurnTask
            processed_vision_data = current_task.marker_vision_data
        # Add similar checks if other tasks store specific vision results
        # ---

        if status == TaskStatus.COMPLETED and self.current_task_index < len(self.mission_plan) - 1:
            # (Task transition logic remains the same)
            # Remove SlalomTask specific logic if SlalomTask is no longer used
            # if isinstance(current_task, SlalomTask) and not current_task.reversed:
            #     ...
            self.current_task_index += 1
            next_task = self.mission_plan[self.current_task_index]
            if hasattr(next_task, 'on_start'):
                # Pass sensors, as some on_start might need it (like Stabilize implicitly)
                next_task.on_start(self, sensors)
            # Remove HoverTask specific logic if Stabilize is always used
            # if isinstance(next_task, HoverTask):
            #     self.target_x, self.target_y = sensors.x, sensors.y

        # Return commands and the *processed* vision data for debug drawing
        return commands, processed_vision_data

    # (Helper methods like get_current_task_name, get_current_state_name,
    #  _mix_and_normalize_commands, get_heading_commands, _get_pid_hover_commands,
    #  _get_damping_commands, get_go_to_visual_target_commands remain the same
    #  as the last version provided)
    # ... [rest of the helper methods] ...

    def get_current_task_name(self) -> str:
        if self.current_task_index < len(self.mission_plan):
             return self.mission_plan[self.current_task_index].__class__.__name__
        return "MISSION_COMPLETE"

    def get_current_state_name(self) -> str:
        if self.current_task_index < len(self.mission_plan):
             return self.mission_plan[self.current_task_index].state_name
        return ""

    def _get_navigation_target(self, vision_data: VisionData, cam_w: int) -> Tuple[float | None, str | None]:
        # This might become less relevant if AlignToObjectX handles it
        l, r = vision_data.left_passage_center_x, vision_data.right_passage_center_x
        if l and r:
             return (l, 'left') if abs(l - cam_w/2) < abs(r - cam_w/2) else (r, 'right')
        return (l, 'left') if l else ((r, 'right') if r else (None, None))

    def get_search_commands(self, sensors, target_heading=None, surge_power=0.0):
        yaw = self.SEARCH_TURN_COMMAND
        if target_heading:
             yaw_error = angle_diff(target_heading, sensors.heading)
             yaw_command = yaw_error * self.HOVER_YAW_P_GAIN
             yaw = np.clip(yaw_command, -self.SCAN_TURN_COMMAND, self.SCAN_TURN_COMMAND)
        return self._mix_and_normalize_commands(surge_power, 0, yaw)

    def get_spin_damping_commands(self, sensors: SensorSuite):
        h_rad, cos_h, sin_h = math.radians(sensors.heading), math.cos(math.radians(sensors.heading)), math.sin(math.radians(sensors.heading))
        wd_x, wd_y = -sensors.velocity_x * self.ALIGN_DAMPING_GAIN, -sensors.velocity_y * self.ALIGN_DAMPING_GAIN
        surge, sway = wd_x * cos_h + wd_y * sin_h, -wd_x * sin_h + wd_y * cos_h
        yaw = -sensors.imu.gyro_z * self.YAW_D_GAIN
        return self._mix_and_normalize_commands(surge, sway, yaw)

    def get_heading_commands(self, sensors, heading, surge_power=0.0):
        yaw = angle_diff(heading, sensors.heading) * self.HOVER_YAW_P_GAIN
        return self._mix_and_normalize_commands(surge_power, 0, yaw)

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
        yaw_cmd = angle_diff(self.target_heading, sensors.heading) * yaw_p_gain
        surge = fsh
        return self._mix_and_normalize_commands(surge, sway, yaw_cmd)

    def _get_damping_commands(self, sensors: SensorSuite) -> ThrusterCommands:
        wdx, wdy = -sensors.velocity_x * self.DAMPING_GAIN, -sensors.velocity_y * self.DAMPING_GAIN
        h_rad = math.radians(sensors.heading)
        c, s = math.cos(h_rad), math.sin(h_rad)
        fsh, sway = wdx*c+wdy*s, -wdx*s+wdy*c
        yaw = angle_diff(self.target_heading, sensors.heading) * self.HOVER_YAW_P_GAIN
        surge = fsh
        return self._mix_and_normalize_commands(surge, sway, yaw)

    def _mix_and_normalize_commands(self, surge, sway, yaw) -> ThrusterCommands:
        commands = ThrusterCommands()
        commands.port = surge + yaw
        commands.starboard = surge - yaw
        max_abs = max(1.0, abs(commands.port), abs(commands.starboard))
        if max_abs > 1.0:
             commands.port /= max_abs
             commands.starboard /= max_abs
        return commands

    def get_go_to_visual_target_commands(self, sensors: SensorSuite, nav_target_x: float, surge_power: float):
        cam_w, cam_h = sensors.camera_image.get_size()
        pixel_error_x = nav_target_x - (cam_w / 2)
        yaw_p = -(pixel_error_x / (cam_w / 2)) * self.ALIGN_YAW_P_GAIN
        yaw_d = -sensors.imu.gyro_z * self.YAW_D_GAIN
        yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)
        h_rad = math.radians(sensors.heading)
        sway = (sensors.velocity_x * math.sin(h_rad) - sensors.velocity_y * math.cos(h_rad)) * self.MANEUVER_DAMPING_GAIN
        return self._mix_and_normalize_commands(surge_power, sway, yaw)