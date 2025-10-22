#!/usr/bin/env python3
"""
Main entry point for the Autonomous Submarine Simulator.
Assembles and starts the simulation.
"""
from simulator import SubmarineSimulator
from ai.submarine import Submarine
# --- Absolute imports from ai.tasks package ---
# --- MODIFY IMPORTS ---
from ai.tasks import GateTask, StabilizeTask, RatchetTurnTask # Was MarkerTurnTask
# ---

if __name__ == "__main__":
    MISSION_DEPTH = 0.1 # Run on the surface

    # --- Create task instances ---
    first_gate_task = GateTask(target_depth=MISSION_DEPTH)
    stabilize_after_gate1 = StabilizeTask(duration=2.0)
    ratchet_task = RatchetTurnTask(initial_target_fraction=0.6, 
                                   subsequent_target_fraction=0.95, 
                                   surge_power=0.25, 
                                   yaw_power=-0.3) 
    stabilize_after_turn = StabilizeTask(duration=2.0)
    return_gate_task = GateTask(target_depth=MISSION_DEPTH)

    # --- Set initial search direction for RatchetTurnTask ---
    ratchet_task.search_direction = -1 # Initial search right for pole

    # --- Set search direction for the return GateTask ---
    return_gate_task.search_direction = -1 # Search right for gate after turn
    # ---

    # --- Define mission plan using the instances ---
    mission = [
        first_gate_task,
        stabilize_after_gate1,
        ratchet_task,
        stabilize_after_turn,
        return_gate_task 
    ]

    submarine_ai = Submarine(mission_plan=mission)
    sim = SubmarineSimulator(submarine_ai=submarine_ai)
    sim.run()