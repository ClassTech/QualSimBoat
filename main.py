#!/usr/bin/env python3
"""
Main entry point for the Autonomous Surface Vehicle (ASV) Simulator.
Assembles and starts the simulation.
"""
from simulator import SubmarineSimulator
from ai.submarine import Submarine
# --- Absolute imports from ai.tasks package ---
from ai.tasks import GateTask, StabilizeTask
# Import the base Task class and common subtasks
from ai.tasks.task_base import Task
from ai.tasks.common_subtasks import (WaitForTargetVisible, 
                                      OrbitUsingSideCamera, TurnToHeading, AlignUsingSideCamera,
                                      Stabilize, ApproachObjectByWidth) # Added new imports
# ---

if __name__ == "__main__":
    MISSION_DEPTH = 0.1 # Run on the surface

    # --- Create task instances ---
    first_gate_task = GateTask(target_depth=MISSION_DEPTH)
    stabilize_after_gate1 = StabilizeTask(duration=2.0) # This is a Task, this is correct

    # --- Side orbit task definition ---
    side_orbit_task = Task()
    side_orbit_task.subtasks = [
        # 1. Find the pole with the front camera
        WaitForTargetVisible(target_type='pole'),

        # 2. Approach by WIDTH
        ApproachObjectByWidth(
            target_width_px=30,     # Approach until pole is 35px wide (Tune this!)
            width_tolerance_px=5,   # Tune this!
            surge_p_gain=0.05,      # Tune this!
            yaw_gain=1.5
        ),

        # 3. Stop all motion
        Stabilize(duration=1.0, speed_threshold=0.05),

        # 4. Yaw RIGHT until pole is centered in the SIDE camera
        AlignUsingSideCamera(
            target_x_fraction=0.5,      # Center of the side camera
            tolerance_px=20,            # Reset tolerance back
            search_yaw_power=0.15,      # Keep slower search (Yaw RIGHT)
            align_yaw_gain=1.5,         # Keep strong positive gain
            yaw_rate_tolerance=0.05
        ),

        # 5. Orbit using side camera until gate is spotted in front camera
        OrbitUsingSideCamera(
            target_pole_width_px=40,    # Target distance (40px wide). Tune this!
            orbit_surge_power=0.25,     # Was 0.05
            yaw_x_gain=0.025,            # Positive gain for centering. Was 0.015
            yaw_dist_p_gain=-0.3,       # Was -0.3
            # --- ADDED: Integral gain for distance ---
            yaw_dist_i_gain=-1.3,      # Was -0.5
            integral_clamp = 2.5    ,    # Was 1.0 
            # completion_gate_tolerance_px is no longer used
        )
    ]
    # Set initial search direction for this task
    side_orbit_task.search_direction = -1 # Initial search right for pole
    # ---

    stabilize_after_turn = StabilizeTask(duration=2.0) # This Task might be redundant now, but keep for testing
    return_gate_task = GateTask(target_depth=MISSION_DEPTH)

    # --- Set search direction for the return GateTask ---
    return_gate_task.search_direction = -1 # Search right for gate after turn
    # ---

    # --- Define mission plan using the new task ---
    mission = [
        first_gate_task,
        stabilize_after_gate1,
        side_orbit_task,        # <-- USE THE NEW TASK
        stabilize_after_turn,
        return_gate_task
    ]

    submarine_ai = Submarine(mission_plan=mission)
    sim = SubmarineSimulator(submarine_ai=submarine_ai)

    # --- Helper code to catch debug drawing errors (no changes needed here) ---
    old_run = sim.run
    def new_run(self):
        try:
            old_run()
        except TypeError as e:
            if "'tuple' object has no attribute 'red_blobs'" in str(e) or "'tuple' object has no attribute 'get_gate_pair'" in str(e):
                print("\n---")
                print("FIX: Your simulator.py debug drawing logic needs to be updated.")
                print("In sim.run(), change 'vision_results.red_blob' to 'vision_results[0].red_blobs'")
                print("Do this for .get_gate_pair(), .get_best_pole() etc. (Use vision_results[0] for the front camera)")
                print("---\n")
            raise e
    sim.run = new_run.__get__(sim, SubmarineSimulator)
    # ---

    sim.run()