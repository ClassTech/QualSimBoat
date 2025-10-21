# Autonomous Surface Vehicle (ASV) Simulator - Pre-Qualification Course

## Description

This project simulates an Autonomous Surface Vehicle (ASV) performing a pre-qualification course consisting of navigating through a gate and maneuvering around a marker pole. It was originally based on the [ClassTech/RoboSim](https://github.com/ClassTech/RoboSim) project for RoboSub 2025 tasks but has been significantly modified.

The simulator uses **Pygame** for 2D top-down map visualization and provides a **simulated 3D camera view** from the ASV's perspective. The ASV operates entirely on the surface (at Z=0) and uses a differential drive (two thrusters) for propulsion and steering. The physics model includes distinct surge (forward/backward) and sway (sideways) drag coefficients to better represent boat-like movement.

---

## Features

* **2D Map View:** Top-down visualization of the ASV and course elements.
* **3D Camera Simulation:** Renders a first-person view from the ASV, including course elements extending above and below the water.
* **ASV Physics Model:** Simulates a differential drive boat with distinct surge, sway, and angular drag, as well as rotational inertia. Locked to operate at Z=0.
* **Modular AI (Task/Subtask Architecture):**
    * **Tasks:** Define high-level mission goals (e.g., `GateTask`, `MarkerTurnTask`). They manage a sequence of subtasks and handle specific vision processing.
    * **Subtasks:** Define reusable, atomic actions (e.g., `AlignToObjectX`, `DriveStraight`, `TurnToHeading`, `Stabilize`, `WaitForTargetVisible`). Subtasks communicate via a shared `context` dictionary managed by the parent Task.
    * **Failure Handling:** Tasks can define recovery behavior (like restarting) when a subtask fails.
* **Computer Vision Simulation:** Uses OpenCV (`cv2`) via the `ai.vision.find_blobs_hsv` function to simulate blob detection based on HSV color ranges defined in `config.py`.
* **Configurable:** Key parameters (world dimensions, physics coefficients, colors, AI tuning gains, subtask parameters) are defined in `config.py`.

---

## Current Course (Pre-Qualification)

The simulated environment contains:

1.  **Gate:** A horizontal opening (2m wide) marked by two vertical gray poles extending 2ft (0.61m) above the water surface and down to the pool floor. The traversable opening (red outline in 3D view) is submerged 1m below the surface.
2.  **Marker:** A vertical green pole located 10m beyond the gate, extending from the floor to 2ft (0.61m) above the water surface.

The current mission objective (`main.py`) is:
1.  Pass through the gate (`GateTask`).
2.  Stabilize briefly (`Stabilize`).
3.  Navigate around the green marker pole using an offset approach and orbit (`MarkerTurnTask`).
4.  Stabilize briefly (`Stabilize`).
5.  Return through the gate (`GateTask`).

---

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install pygame numpy opencv-python
    ```
    *(Ensure you have Python 3 installed. Tested with Python 3.12/3.13.)*

---

## Usage

Run the main simulation script from the root directory:

```bash
python main.py