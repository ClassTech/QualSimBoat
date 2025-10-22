#!/usr/bin/env python3
"""
Defines simple data classes used for passing information between modules,
including the Vision class which holds an image provider and analyzes on demand.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any, Callable # Added Callable
import pygame
import numpy as np 
# --- Import the blob finding function ---
from ai.vision import find_blobs_hsv
# ---
# --- Import HSV Ranges (needed for thresholds potentially) ---
from config import GREEN_HSV_RANGE, RED_HSV_RANGES, SimulationConfig # Added SimConfig if needed later
# ---

@dataclass
class MPU6050Readings:
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0
    gyro_z: float = 0.0

@dataclass
class SensorSuite:
    camera_image: pygame.Surface
    depth: float
    heading: float
    pitch: float
    imu: MPU6050Readings
    x: float = 0.0
    y: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    angular_velocity_y: float = 0.0
    velocity_z: float = 0.0

# --- UPDATED Vision Class ---
class Vision:
    """
    Handles vision processing. Stores an image provider, analyzes the image 
    when update() is called, and provides methods to interpret results.
    """
    def __init__(self, 
                 image_provider: Callable[[], pygame.Surface],
                 # Default thresholds - consider passing these from Submarine/Config
                 min_pole_pixels: int = 20, 
                 min_gate_pixels: int = 50):
                 
        if not callable(image_provider):
             raise TypeError("image_provider must be a callable function")
             
        self.image_provider = image_provider
        self.min_pole_pixels = min_pole_pixels
        self.min_gate_pixels = min_gate_pixels

        # --- Internal storage for last analysis ---
        self.green_blobs: List[Dict] = []
        self.red_blobs: List[Dict] = []
        # ---

        # --- Cached results (remain the same structure) ---
        self._clear_cache()

    def _clear_cache(self):
        """ Resets cached interpretations. Called by update()."""
        self._best_green_pole: Optional[Dict] = None
        self._found_best_green_pole: bool = False 
        
        self._best_gate_pair: Optional[Tuple[Dict, Dict]] = None
        self._found_best_gate_pair: bool = False 
        
    def update(self): # Renamed from analyze()
        """ 
        Fetches the current camera image, performs blob detection for
        relevant colors, stores results internally, and clears caches.
        """
        current_image = self.image_provider()
        if current_image is None:
             # Handle case where provider fails (e.g., during shutdown)
             self.green_blobs = []
             self.red_blobs = []
        else:
             # Perform Blob Detection
             self.green_blobs = find_blobs_hsv(current_image, GREEN_HSV_RANGE, self.min_pole_pixels)
             self.red_blobs = find_blobs_hsv(current_image, RED_HSV_RANGES, self.min_gate_pixels)
        
        # Clear any previous interpretations
        self._clear_cache()

    # --- Interpretation Methods (operate on self.green_blobs / self.red_blobs) ---
    # (These methods remain identical to the previous VisionResults version)
    
    # --- Methods for Green Pole (Marker) ---
    def get_best_pole(self) -> Optional[Dict]:
        if not self._found_best_green_pole:
            self._found_best_green_pole = True 
            if not self.green_blobs: self._best_green_pole = None
            else:
                potential_poles = [b for b in self.green_blobs if b['height'] > b['width'] * 1.2] 
                if not potential_poles: self._best_green_pole = None
                else: self._best_green_pole = max(potential_poles, key=lambda b: b['area'])
        return self._best_green_pole

    def is_pole_visible(self) -> bool: return self.get_best_pole() is not None
    def get_pole_center_x(self) -> Optional[float]:
        pole = self.get_best_pole(); return pole['center_x'] if pole else None
    def get_pole_apparent_height(self) -> float:
        pole = self.get_best_pole(); return pole['height'] if pole else 0.0

    # --- Methods for Red Poles (Gate) ---
    def get_gate_pair(self) -> Optional[Tuple[Dict, Dict]]:
        if not self._found_best_gate_pair:
            self._found_best_gate_pair = True
            if len(self.red_blobs) < 2: self._best_gate_pair = None
            else:
                potential_poles = [b for b in self.red_blobs if b['height'] > b['width'] * 1.5] 
                if len(potential_poles) < 2: self._best_gate_pair = None
                else:
                    potential_poles.sort(key=lambda p: p['center_x'])
                    best_pair_found, min_height_diff = None, float('inf')
                    for i in range(len(potential_poles) - 1):
                        p1, p2 = potential_poles[i], potential_poles[i+1]
                        height_diff = abs(p1['height'] - p2['height'])
                        y_overlap = min(p1['max_y'], p2['max_y']) - max(p1['min_y'], p2['min_y'])
                        if y_overlap > 10 and height_diff < 150: 
                             if height_diff < min_height_diff:
                                 min_height_diff = height_diff
                                 best_pair_found = (p1, p2)
                    self._best_gate_pair = best_pair_found 
        return self._best_gate_pair

    def is_gate_visible(self) -> bool: return self.get_gate_pair() is not None
    def get_gate_center_x(self) -> Optional[float]:
        pair = self.get_gate_pair(); return (pair[0]['center_x'] + pair[1]['center_x']) / 2 if pair else None

# Simplified ThrusterCommands (remains the same)
@dataclass
class ThrusterCommands:
    port: float = 0.0
    starboard: float = 0.0
    pause_simulation: bool = False