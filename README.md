# Autonomous Surface Vehicle (ASV) Simulator - Pre-Qualification Course

## Description

This project simulates an Autonomous Surface Vehicle (ASV) performing a pre-qualification course consisting of navigating through a gate and maneuvering around a marker pole. It was originally based on the [ClassTech/RoboSim](https://github.com/ClassTech/RoboSim) project for RoboSub 2025 tasks but has been significantly modified.

The simulator uses **Pygame** for 2D top-down map visualization and provides **simulated 3D camera views** from the ASV's perspective (both front-facing and right-side). The ASV operates entirely on the surface (at Z=0) and uses a differential drive (two thrusters) for propulsion and steering. The physics model includes distinct surge (forward/backward) and sway (sideways) drag coefficients to better represent boat-like movement.

---

## Features

* **2D Map View:** Top-down visualization of the ASV and course elements.
* **Simulated 3D Cameras:** Renders first-person views from the ASV's front and right (starboard) side, including course elements extending above and below the water.
* **ASV Physics Model:** Simulates a differential drive boat with distinct surge, sway, and angular drag, as well as rotational inertia. Locked to operate at Z=0. The sway drag coefficient can be tuned in `config.py` to adjust sideways drift realism.
* **Modular AI (Task/Subtask Architecture):**
    * **Tasks:** Define high-level mission goals (e.g., `GateTask`, `side_orbit_task`). They manage a sequence of subtasks.
    * **Subtasks:** Define reusable, atomic actions (e.g., `AlignToObjectX`, `DriveStraight`, `Stabilize`, `ApproachObjectByWidth`, `AlignUsingSideCamera`, `OrbitUsingSideCamera`). Subtasks communicate via a shared `context` dictionary managed by the parent Task.
* **Computer Vision Simulation:** Uses OpenCV (`cv2`) via the `ai.vision.find_blobs_hsv` function to simulate blob detection based on HSV color ranges defined in `config.py`. The `Vision` class (in `data_structures.py`) processes images from both cameras.
* **Configurable:** Key parameters (world dimensions, physics coefficients, colors, AI tuning gains, subtask parameters) are defined in `config.py`.

---

## Current Course (Pre-Qualification)

The simulated environment contains:

1.  **Gate:** A horizontal opening (2m wide) marked by two vertical red poles extending 0.61m above the water surface and down to the pool floor.
2.  **Marker:** A vertical green pole located 10m beyond the gate, extending from the floor to 0.61m above the water surface.

The current mission objective (`main.py`) is:
1.  Pass through the gate using `GateTask` (actively steers towards center).
2.  Stabilize briefly (`StabilizeTask`).
3.  Execute the `side_orbit_task`:
    * Find the green marker pole using the front camera (`WaitForTargetVisible`).
    * Approach the pole to a set distance based on its visual width in the front camera (`ApproachObjectByWidth`).
    * Stop (`Stabilize`).
    * Yaw right until the pole is centered in the side camera (`AlignUsingSideCamera`).
    * Orbit the pole using the side camera, maintaining distance via pole width and keeping it centered, while driving forward slowly. Stop orbiting when the return gate is spotted in the front camera (`OrbitUsingSideCamera`).
    * Align to the center of the return gate using the front camera (`AlignToObjectX`).
4.  Stabilize briefly (`StabilizeTask`).
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
    *(Ensure you have Python 3 installed. Tested with Python 3.12.)*

---

## Usage

Run the main simulation script from the root directory:

```bash
python main.py