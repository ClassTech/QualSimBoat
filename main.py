#!/usr/bin/env python3
"""
Main entry point for the Autonomous Submarine Simulator.
Assembles and starts the simulation.
"""
from simulator import SubmarineSimulator
from ai.submarine import Submarine
# --- Absolute imports from ai.tasks package ---
from ai.tasks import GateTask, StabilizeTask, MarkerTurnTask
# ---

if __name__ == "__main__":
    MISSION_DEPTH = 0.1 # Run on the surface

    mission = [
        GateTask(target_depth=MISSION_DEPTH),
        StabilizeTask(duration=2.0),
        MarkerTurnTask(target_depth=MISSION_DEPTH),
        StabilizeTask(duration=2.0),
        GateTask(target_depth=MISSION_DEPTH)
    ]

    submarine_ai = Submarine(mission_plan=mission)
    sim = SubmarineSimulator(submarine_ai=submarine_ai)
    sim.run()