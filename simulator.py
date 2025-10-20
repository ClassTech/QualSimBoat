#!/usr/bin/env python3
"""
Contains the main SubmarineSimulator class.
This class handles Pygame, rendering, physics, and the main game loop.
"""
import math
import random
import time
from typing import Tuple, Optional

import pygame
import numpy as np

from config import *
from world import SubmarinePhysicsState
from world import PrequalGate, PrequalMarker
from data_structures import ThrusterCommands, MPU6050Readings, SensorSuite
from ai.submarine import Submarine


class SubmarineSimulator:
    def __init__(self, submarine_ai: Submarine, width=1200, height=800):
        pygame.init()
        self.width, self.height = width, height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Autonomous Surface Vehicle (ASV) Simulator")
        self.clock = pygame.time.Clock()
        self.config = SimulationConfig()
        self.prequal_config = PrequalConfig()
        self.scaleX = width * 0.7 / self.config.worldWidth
        self.scaleY = height * 0.8 / self.config.worldHeight
        self.font = pygame.font.Font(None, 36)
        self.smallFont = pygame.font.Font(None, 24)
        self.cameraSurface = pygame.Surface((320, 240))
        try:
            # (Image loading remains the same)
            bg_img = pygame.image.load("image_9c266f.jpg").convert()
            h=480; w=int(bg_img.get_width()*(h/bg_img.get_height()))
            self.camera_background = pygame.transform.scale(bg_img, (w,h))
            self.camera_background_pano = pygame.Surface((w*2,h))
            self.camera_background_pano.blit(self.camera_background,(0,0)); self.camera_background_pano.blit(self.camera_background,(w,0))
        except pygame.error: self.camera_background, self.camera_background_pano = None, None

        # --- Physics Properties ---
        self.subMass = 4.0 # Keep estimated mass
        # --- MODIFIED: Increased Inertia ---
        self.subInertia = 0.35 # Was 0.15 - More resistance to starting/stopping turns
        # ---
        self.netBuoyancyForce, self.thrusterMaxForce = 0.0, 0.8
        self.surgeDragCoeff = 1.5 # Keep surge drag
        self.swayDragCoeff = 8.0  # Keep high sway drag
        # --- MODIFIED: Increased Angular Drag ---
        self.angularDragCoeff = 0.25 # Was 0.1 - More resistance to turning
        # ---

        self.submarineAI = submarine_ai
        self.prequal_gate: Optional[PrequalGate] = None
        self.prequal_marker: Optional[PrequalMarker] = None
        self.resetSimulation()

    def resetSimulation(self):
        # (resetSimulation remains the same)
        self.prequal_gate = PrequalGate(
            x = self.prequal_config.GATE_X_POS, center_y = self.config.worldHeight / 2,
            z_top = self.prequal_config.GATE_DEPTH_METERS, width = self.prequal_config.GATE_WIDTH_METERS,
            height = self.prequal_config.GATE_OPENING_HEIGHT, color = self.prequal_config.GATE_COLOR
        )
        self.prequal_marker = PrequalMarker(
            x = self.prequal_config.MARKER_X_POS, y = self.config.worldHeight / 2,
            z_top = -self.prequal_config.POLE_ABOVE_SURFACE_METERS,
            z_bottom = self.config.worldDepth - 0.01,
            radius = self.prequal_config.MARKER_DIAMETER_METERS / 2, color = self.prequal_config.MARKER_COLOR
        )
        start_heading = 0
        self.subPhysics = SubmarinePhysicsState(
            x = self.prequal_config.START_X_POS, y = self.config.worldHeight / 2 + random.uniform(-0.5, 0.5),
            z = 0.0, heading = start_heading, pitch = 0.0
        )
        self.submarineAI.reset()
        self.startTime = time.time()
        self.running, self.paused = True, False
        self.lastThrusterCommands, self.last_imu_readings = ThrusterCommands(), MPU6050Readings()


    def worldToScreen(self, x, y):
        # (Remains the same)
        return int(x*self.scaleX+50), int((self.config.worldHeight-y)*self.scaleY+50)

    def handleInput(self):
        # (Remains the same)
        for event in pygame.event.get():
             if event.type == pygame.QUIT: self.running = False
             elif event.type == pygame.KEYDOWN:
                 if event.key == pygame.K_SPACE: self.paused = not self.paused
                 elif event.key == pygame.K_r: self.resetSimulation()

    def applyPhysics(self, dt, commands):
        # (applyPhysics with separate surge/sway drag remains the same)
        f_port = commands.port * self.thrusterMaxForce
        f_starboard = commands.starboard * self.thrusterMaxForce
        thrust_surge = f_port + f_starboard
        thrust_sway = 0.0
        thrust_yaw = (f_port - f_starboard) * (self.config.submarineWidth / 2)
        h_rad = math.radians(self.subPhysics.heading)
        cos_h, sin_h = math.cos(h_rad), math.sin(h_rad)
        vel_surge = self.subPhysics.velocity_x * cos_h + self.subPhysics.velocity_y * sin_h
        vel_sway = -self.subPhysics.velocity_x * sin_h + self.subPhysics.velocity_y * cos_h
        drag_surge = -self.surgeDragCoeff * vel_surge * abs(vel_surge)
        drag_sway = -self.swayDragCoeff * vel_sway * abs(vel_sway)
        drag_yaw = -self.angularDragCoeff * self.subPhysics.angular_velocity_z**2 * np.sign(self.subPhysics.angular_velocity_z)
        total_force_surge = thrust_surge + drag_surge
        total_force_sway = thrust_sway + drag_sway
        total_torque_yaw = thrust_yaw + drag_yaw
        fx = total_force_surge * cos_h - total_force_sway * sin_h
        fy = total_force_surge * sin_h + total_force_sway * cos_h
        ax, ay = fx / self.subMass, fy / self.subMass
        angular_accel_z = total_torque_yaw / self.subInertia
        self.subPhysics.velocity_x += ax * dt
        self.subPhysics.velocity_y += ay * dt
        self.subPhysics.angular_velocity_z += angular_accel_z * dt
        self.subPhysics.velocity_z = 0.0
        self.subPhysics.angular_velocity_y = 0.0
        imu_accel_surge = ax * cos_h + ay * sin_h
        imu_accel_sway = -ax * sin_h + ay * cos_h
        self.last_imu_readings = MPU6050Readings(
            accel_x=imu_accel_sway, accel_y=imu_accel_surge, accel_z=0.0,
            gyro_z=self.subPhysics.angular_velocity_z
        )
        self.subPhysics.x += self.subPhysics.velocity_x * dt
        self.subPhysics.y += self.subPhysics.velocity_y * dt
        self.subPhysics.heading = (self.subPhysics.heading + math.degrees(self.subPhysics.angular_velocity_z * dt)) % 360
        self.subPhysics.z = 0.0
        self.subPhysics.pitch = 0.0
        margin = 0.5
        self.subPhysics.x = np.clip(self.subPhysics.x, margin, self.config.worldWidth - margin)
        self.subPhysics.y = np.clip(self.subPhysics.y, margin, self.config.worldHeight - margin)

    def project3D(self, world_pos: Tuple[float, float, float]) -> Optional[Tuple[int, int, float]]:
        # (Remains the same)
        dx,dy,dz = world_pos[0]-self.subPhysics.x, world_pos[1]-self.subPhysics.y, world_pos[2]-self.subPhysics.z
        h,p = math.radians(-self.subPhysics.heading), math.radians(-self.subPhysics.pitch)
        ch,sh,cp,sp = math.cos(h),math.sin(h),math.cos(p),math.sin(p)
        x_yaw, y_yaw = dx*ch-dy*sh, dx*sh+dy*ch
        cz,cy,cx = x_yaw*cp+dz*sp, x_yaw*sp-dz*cp, y_yaw
        if cz < 0.2: return None
        w,h = self.cameraSurface.get_size()
        f = w/(2*math.tan(math.radians(self.config.cameraFov/2)))
        return int(w/2-f*(cx/cz)), int(h/2-f*(cy/cz)), math.hypot(dx,dy,dz)


    def generateCameraView(self):
        # (Remains the same)
        w,h = self.cameraSurface.get_size()
        if self.camera_background_pano:
             bg_w,bg_h = self.camera_background.get_size()
             x_off = (self.subPhysics.heading/360)*bg_w
             y_off = np.clip(((bg_h-h)/2)-(self.subPhysics.pitch*2.0), 0, bg_h-h)
             self.cameraSurface.blit(self.camera_background_pano, (-x_off, -y_off))
        else:
             self.cameraSurface.fill(WATER_COLOR)
             hp = self.project3D((self.subPhysics.x+20, self.subPhysics.y, self.config.worldDepth))
             if hp: pygame.draw.rect(self.cameraSurface, POOL_FLOOR_COLOR, (0,hp[1],w,h))
        drawable = []
        if self.prequal_gate:
             g = self.prequal_gate; half_w = g.width / 2; z_bottom = g.z_top + g.height
             corners_3d = [(g.x, g.center_y - half_w, g.z_top), (g.x, g.center_y + half_w, g.z_top),(g.x, g.center_y + half_w, z_bottom), (g.x, g.center_y - half_w, z_bottom)]
             proj_corners = [self.project3D(p) for p in corners_3d]
             if all(proj_corners): points_2d = [p[:2] for p in proj_corners]; avg_dist = sum(p[2] for p in proj_corners) / 4; drawable.append((avg_dist, 'polygon', g.color, points_2d, 5))
             pole_z_top = -self.prequal_config.POLE_ABOVE_SURFACE_METERS; pole_z_bottom = self.config.worldDepth - 0.01; pole_color = GRAY
             lp_top = self.project3D((g.x, g.center_y - half_w, pole_z_top)); lp_bot = self.project3D((g.x, g.center_y - half_w, pole_z_bottom))
             if lp_top and lp_bot: avg_dist = (lp_top[2] + lp_bot[2]) / 2; drawable.append((avg_dist, 'line', pole_color, lp_top[:2], lp_bot[:2], 5))
             rp_top = self.project3D((g.x, g.center_y + half_w, pole_z_top)); rp_bot = self.project3D((g.x, g.center_y + half_w, pole_z_bottom))
             if rp_top and rp_bot: avg_dist = (rp_top[2] + rp_bot[2]) / 2; drawable.append((avg_dist, 'line', pole_color, rp_top[:2], rp_bot[:2], 5))
        if self.prequal_marker:
             m = self.prequal_marker; tp = self.project3D((m.x, m.y, m.z_top)); bp = self.project3D((m.x, m.y, m.z_bottom))
             if tp and bp: avg_dist = (tp[2] + bp[2]) / 2; drawable.append((avg_dist, 'line', m.color, tp[:2], bp[:2], 8))
        drawable.sort(key=lambda x: x[0], reverse=True)
        for d in drawable:
             if d[1]=='polygon': pygame.draw.polygon(self.cameraSurface, d[2], d[3], d[4])
             elif d[1]=='line': pygame.draw.line(self.cameraSurface, d[2], d[3], d[4], d[5])
             elif d[1]=='rect': pygame.draw.rect(self.cameraSurface, d[2], d[3])


    def render(self):
        # (Remains the same)
        self.screen.fill(LIGHT_BLUE)
        pygame.draw.rect(self.screen, BLACK, (40,40,int(self.config.worldWidth*self.scaleX+20),int(self.config.worldHeight*self.scaleY+20)), 2)
        if self.prequal_gate: g = self.prequal_gate; p1 = self.worldToScreen(g.x, g.center_y - g.width / 2); p2 = self.worldToScreen(g.x, g.center_y + g.width / 2); pygame.draw.line(self.screen, g.color, p1, p2, 4)
        if self.prequal_marker: m = self.prequal_marker; pos_2d = self.worldToScreen(m.x, m.y); radius_scaled = max(2, int(m.radius * self.scaleX)); pygame.draw.circle(self.screen, m.color, pos_2d, radius_scaled); pygame.draw.circle(self.screen, BLACK, pos_2d, radius_scaled, 1)
        subPos = self.worldToScreen(self.subPhysics.x, self.subPhysics.y); hRad, cos_h, sin_h = math.radians(self.subPhysics.heading), math.cos(math.radians(self.subPhysics.heading)), math.sin(math.radians(self.subPhysics.heading)); pvc_s = self.config.submarineWidth*self.scaleX/2
        corners = [(-pvc_s,-pvc_s), (pvc_s,-pvc_s), (pvc_s,pvc_s), (-pvc_s,pvc_s)]; rotated = [(subPos[0]+dx*cos_h-dy*sin_h, subPos[1]-(dx*sin_h+dy*cos_h)) for dx,dy in corners]; pygame.draw.polygon(self.screen,YELLOW,rotated,4)
        box_w,box_l=0.127*self.scaleY/2,self.config.submarineLength*self.scaleX/2; box_corners=[(-box_l,-box_w),(box_l,-box_w),(box_l,box_w),(-box_l,box_w)]; rotated_box=[(subPos[0]+dx*cos_h-dy*sin_h,subPos[1]-(dx*sin_h+dy*cos_h)) for dx,dy in box_corners]; pygame.draw.polygon(self.screen,CONTROL_BOX_GRAY,rotated_box)
        arrow_pts = [(box_l,-box_w),(box_l,box_w),(box_l+0.2*self.scaleX,0)]; rotated_arrow=[(subPos[0]+dx*cos_h-dy*sin_h, subPos[1]-(dx*sin_h+dy*cos_h)) for dx,dy in arrow_pts]; pygame.draw.polygon(self.screen, YELLOW, rotated_arrow)
        self._renderUi(); scaled_camera = pygame.transform.scale(self.cameraSurface, (400, 300)); self.screen.blit(scaled_camera, (self.width-420, 20)); pygame.draw.rect(self.screen, BLACK, (self.width-420, 20, 400, 300), 2); pygame.display.flip()

    def _drawThrusterBar(self, x, y, label, value):
        # (Remains the same)
        bar_w, bar_h = 25, 80; center_y = y + bar_h / 2; pygame.draw.rect(self.screen, GRAY, (x, y, bar_w, bar_h), 2)
        if value > 0: pygame.draw.rect(self.screen, GREEN, (x + 1, center_y - value * bar_h / 2, bar_w - 2, value * bar_h / 2))
        elif value < 0: pygame.draw.rect(self.screen, RED, (x + 1, center_y, bar_w - 2, abs(value) * bar_h / 2))
        pygame.draw.line(self.screen, BLACK, (x, center_y), (x + bar_w, center_y), 1); label_surf = self.smallFont.render(label, True, BLACK); self.screen.blit(label_surf, (x + bar_w / 2 - label_surf.get_width() / 2, y + bar_h + 5))


    def _renderUi(self):
        # (Remains the same)
        y = 20; title = self.font.render("Autonomous Surface Vehicle Simulator", True, BLACK); self.screen.blit(title, (20,y)); y+=40
        speed = math.hypot(self.subPhysics.velocity_x, self.subPhysics.velocity_y); task,state = self.submarineAI.get_current_task_name(), self.submarineAI.get_current_state_name()
        stats=[f"Time: {time.time()-self.startTime:.1f}s", f"Task: {task}", f"State: {state}", f"Speed: {speed:.2f} m/s", f"Heading: {self.subPhysics.heading:.1f}°"]
        for s in stats: self.screen.blit(self.smallFont.render(s,True,BLACK),(20,y)); y+=20
        y+=10; imu = self.last_imu_readings
        imu_stats=["IMU:", f" Accel Y(surge): {imu.accel_y: .2f} m/s²", f" Accel X(sway): {imu.accel_x: .2f} m/s²", f" Gyro Z(yaw): {math.degrees(imu.gyro_z): .1f}°/s"]
        for s in imu_stats: self.screen.blit(self.smallFont.render(s,True,BLACK),(20,y)); y+=18
        y = self.height - 80; controls=["Controls:", "R - Reset", "SPACE - Pause"]
        for c in controls: self.screen.blit(self.smallFont.render(c,True,BLACK),(20,y)); y+=18
        tx,ty = self.width-420,350; self.screen.blit(self.smallFont.render("Thruster Output:",True,BLACK),(tx,ty)); ty+=25
        tc=self.lastThrusterCommands; h_labels=[("Port",tc.port),("Star",tc.starboard)]
        for i,(l,v) in enumerate(h_labels): self._drawThrusterBar(tx+i*50,ty,l,v)


    def run(self):
        # (Remains the same)
        while self.running:
             dt = self.clock.tick(60) / 1000.0;
             if dt > 0.1: dt = 0.1
             self.handleInput()
             if self.paused: self.render(); continue
             self.generateCameraView()
             sensors = SensorSuite(camera_image=self.cameraSurface, depth=self.subPhysics.z, heading=self.subPhysics.heading, pitch=self.subPhysics.pitch, imu=self.last_imu_readings, x=self.subPhysics.x, y=self.subPhysics.y, velocity_x=self.subPhysics.velocity_x, velocity_y=self.subPhysics.velocity_y, angular_velocity_y=self.subPhysics.angular_velocity_y, velocity_z=self.subPhysics.velocity_z)
             thrusterCommands, vision_data = self.submarineAI.update(dt, sensors)
             if thrusterCommands.pause_simulation: self.paused = True
             self.lastThrusterCommands = thrusterCommands
             # (Vision debug remains the same)
             for pole in vision_data.potential_poles: pygame.draw.rect(self.cameraSurface, ORANGE, (pole['min_x'], pole['min_y'], pole['max_x']-pole['min_x'], pole['max_y']-pole['min_y']), 1)
             if vision_data.gate_is_visible: w,h = vision_data.max_x-vision_data.min_x, vision_data.max_y-vision_data.min_y; pygame.draw.rect(self.cameraSurface, YELLOW, (vision_data.min_x, vision_data.min_y, w, h), 1)
             for pole in vision_data.visible_poles: w,h = pole['max_x']-pole['min_x'], pole['max_y']-pole['min_y']; color = GREEN if pole.get('color') == 'white' else YELLOW; pygame.draw.rect(self.cameraSurface, color, (pole['min_x'], pole['min_y'], w, h), 1)
             for pole in vision_data.selected_slalom_poles: w = pole['max_x'] - pole['min_x']; h = pole['max_y'] - pole['min_y']; pygame.draw.rect(self.cameraSurface, MAGENTA, (pole['min_x'], pole['min_y'], w, h), 3)
             for pole in vision_data.avoidance_poles: w = pole['max_x'] - pole['min_x']; h = pole['max_y'] - pole['min_y']; pygame.draw.rect(self.cameraSurface, ORANGE, (pole['min_x'], pole['min_y'], w, h), 3)
             self.applyPhysics(dt, thrusterCommands)
             self.render()
        pygame.quit()