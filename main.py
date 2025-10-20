#!/usr/bin/env python3
"""
Main entry point for the Autonomous Submarine Simulator.
Assembles and starts the simulation.
"""
from simulator import SubmarineSimulator
from ai.submarine import Submarine
from ai.tasks import GateTask, StabilizeTask, MarkerTurnTask

if __name__ == "__main__":
    # Define mission-specific parameters in one place
    MISSION_DEPTH = 0.1 # Run on the surface

    # --- MODIFIED: Added StabilizeTask after MarkerTurnTask ---
    mission = [
        # 1. Go through the gate
        GateTask(target_depth=MISSION_DEPTH),

        # 2. Stop and stabilize
        StabilizeTask(duration=2.0),

        # 3. Find, orbit, and turn around the marker
        MarkerTurnTask(target_depth=MISSION_DEPTH),

        # 4. Stop and stabilize again (to kill sway after the turn) <-- NEW
        StabilizeTask(duration=2.0),

        # 5. Go back through the gate
        GateTask(target_depth=MISSION_DEPTH)
    ]
    # --- END MODIFICATION ---

    # 2. Create the Submarine AI engine and give it the mission plan.
    submarine_ai = Submarine(mission_plan=mission)

    # 3. Create the simulator and inject the fully-configured AI.
    sim = SubmarineSimulator(submarine_ai=submarine_ai)

    # 4. Run the simulation.
    sim.run()