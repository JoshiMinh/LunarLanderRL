import math
import numpy as np
import pygame
from pygame import gfxdraw
import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import LunarLander, VIEWPORT_H, VIEWPORT_W, SCALE, FPS
from gymnasium.envs.box2d.lunar_lander import MAIN_ENGINE_POWER, SIDE_ENGINE_POWER, INITIAL_RANDOM
from gymnasium.envs.box2d.lunar_lander import LANDER_POLY, LEG_AWAY, LEG_DOWN, LEG_W, LEG_H, LEG_SPRING_TORQUE
from gymnasium.envs.box2d.lunar_lander import SIDE_ENGINE_HEIGHT, SIDE_ENGINE_AWAY, MAIN_ENGINE_Y_LOCATION
import Box2D
from Box2D.b2 import (
    circleShape,
    edgeShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
)

# Realistic Moon Constants
CUSTOM_VIEWPORT_W = 1280
CUSTOM_VIEWPORT_H = 720
WORLD_W = 5000 / SCALE  # 166 meters
WORLD_H = 3000 / SCALE  # 100 meters
# NASA Lunar Module Style Poly (Collision Mesh)
MODERN_LANDER_POLY = [
    (-8, +18), (+8, +18), (+12, +0), (+14, -18), (-14, -18), (-12, +0)
]
# Detailed visual components
DESCENT_STAGE_POLY = [(-12, -18), (12, -18), (15, -5), (10, 5), (-10, 5), (-15, -5)]
ASCENT_STAGE_POLY = [(-8, 5), (8, 5), (10, 15), (6, 22), (-6, 22), (-10, 15)]
COCKPIT_WINDOW_POLY = [(-4, 15), (4, 15), (5, 18), (-5, 18)]

MOON_RADIUS = 3000 # Visual and structural radius

class VastSpaceLander(LunarLander):
    """
    Ultra-Realistic Scrolling Lunar Descent Simulation.
    Features a modern 'Starship' style lander and a vast surface.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stars = []
        self.camera_x = 0
        self.camera_y = 0
        self.zoom = 1.0
        self.fuel = 300.0 # Tripled fuel
        self.camera_mode = 'focus'
        self.mouse_dragging = False
        self.last_mouse_pos = None
        self.mission_status = None # 'success' or 'failed'
        # Update Observation Space for fuel (9 elements)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32)
        self.generate_cosmos()

    def generate_cosmos(self):
        """Generate distant realistic stars."""
        self.stars = [] # Clear existing if re-generated
        for _ in range(1000): # More stars for wide field
            x = np.random.randint(-CUSTOM_VIEWPORT_W * 10, CUSTOM_VIEWPORT_W * 10) 
            y = np.random.randint(-CUSTOM_VIEWPORT_H * 10, CUSTOM_VIEWPORT_H * 10)
            brightness = np.random.randint(100, 255)
            self.stars.append({'pos': [x, y], 'size': np.random.uniform(0.5, 1.5), 'color': (brightness, brightness, brightness)})

    def reset(self, *, seed=None, options=None):
        self.custom_prev_shaping = None
        self.moon_polys = []
        super().reset(seed=seed, options=options)
        self._rebuild_world_for_realism()
        if self.render_mode == "human":
            self.render()
        return self._get_observation(), {}

    def _rebuild_world_for_realism(self):
        """Build a vast, rugged lunar surface."""
        self._destroy()
        self.gravity = -20.0 # Double gravity for faster descent in the large world
        self.world = Box2D.b2World(gravity=(0, self.gravity))
        from gymnasium.envs.box2d.lunar_lander import ContactDetector
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.custom_prev_shaping = None

        W = WORLD_W
        H = WORLD_H

        # Terrain - 100 chunks for a truly vast surface
        CHUNKS = 100
        height = self.np_random.uniform(H/10, H/4, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        
        # Prepared Landing Pad in a random-ish but central area
        pad_idx = CHUNKS // 2 + self.np_random.integers(-10, 10)
        self.helipad_x1 = chunk_x[pad_idx - 2]
        self.helipad_x2 = chunk_x[pad_idx + 2]
        # Center of moon deep below the pad
        self.moon_center = (W / 2, self.helipad_y - MOON_RADIUS)

        # Robust multi-pass smoothing
        smooth_y = list(height)
        for _ in range(3): 
            new_y = [smooth_y[0]]
            for i in range(1, CHUNKS - 1):
                new_y.append(0.33 * (smooth_y[i-1] + smooth_y[i] + smooth_y[i+1]))
            new_y.append(smooth_y[-1])
            smooth_y = new_y
        
        for i in range(max(0, pad_idx - 5), min(CHUNKS, pad_idx + 6)):
            smooth_y[i] = self.helipad_y

        self.moon = self.world.CreateStaticBody(position=(0, 0))
        self.moon_polys = []
        
        # Geometry follows the curve of the moon circle
        for i in range(CHUNKS - 1):
            x1, y1 = chunk_x[i], smooth_y[i]
            x2, y2 = chunk_x[i+1], smooth_y[i+1]
            
            # Map flat coordinates to spherical arc
            def to_arc(x, h):
                angle = (x - W/2) / MOON_RADIUS
                px = self.moon_center[0] + (MOON_RADIUS + h) * math.sin(angle)
                py = self.moon_center[1] + (MOON_RADIUS + h) * math.cos(angle)
                return px, py

            p1 = to_arc(x1, y1)
            p2 = to_arc(x2, y2)
            
            # For rendering the "underground" slice accurately to follow the curve
            # We'll use more points for the floor to make it look solid
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.5)
            # moon_polys for rendering the surface detail
            self.moon_polys.append([p1, p2, (p2[0], p2[1]-2), (p1[0], p1[1]-2)]) 

        # Create Modern Lander at high altitude
        initial_y = H * 0.85
        initial_x = W / 2 + self.np_random.uniform(-W/10, W/10)
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in MODERN_LANDER_POLY]),
                density=5.0, friction=0.3, categoryBits=0x0010, maskBits=0x001, restitution=0.0,
            ),
        )
        self.fuel = 300.0 # Reset fuel
        self.mission_status = None
        self.lander.ApplyForceToCenter((self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True)

        # Retractable/Modern Legs (Vertical shock absorbers)
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x + i * 12/SCALE, initial_y - 20/SCALE),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(3/SCALE, 8/SCALE)),
                    density=1.0, restitution=0.0, categoryBits=0x0020, maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            rjd = revoluteJointDef(
                bodyA=self.lander, bodyB=leg,
                localAnchorA=(i * 12/SCALE, -20/SCALE), localAnchorB=(0, 8/SCALE),
                enableLimit=True, lowerAngle=-0.1, upperAngle=0.1,
            )
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

    def _get_observation(self):
        """State is relative to Landing Pad."""
        pos = self.lander.position
        vel = self.lander.linearVelocity
        pad_center = (self.helipad_x1 + self.helipad_x2) / 2
        
        # Normalized state
        state = [
            (pos.x - pad_center) / (WORLD_W / 2),
            (pos.y - (self.helipad_y + 20/SCALE)) / (WORLD_H / 2),
            vel.x * 0.2, # Standard LunarLander scale
            vel.y * 0.2, 
            self.lander.angle,
            self.lander.angularVelocity * 0.05,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.fuel / 300.0, # 9th element: Fuel awareness
        ]
        return np.array(state, dtype=np.float32)

    def step(self, action):
        # 0. Fuel Depletion
        if self.fuel <= 0:
            action = 0 
        
        if action == 2: # Main Engine
            self.fuel -= 0.15 # Slower depletion for larger tank
        elif action in [1, 3]: # Side Engines
            self.fuel -= 0.03
        
        self.fuel = max(0, self.fuel)

        # 1. Base physics/particles
        _, _, _, _, info = super().step(action)
        
        # Explicit particle cleanup (Fix: particles were not disappearing)
        for obj in self.particles:
            if hasattr(obj, 'ttl'):
                obj.ttl -= 0.1 # Ensure they eventually hit < 0
        
        # 2. Boost Engine Power
        if self.fuel > 0:
            if action == 2: # Main Engine
                f_v = (0, 12.0 * 5.0) 
                self.lander.ApplyForceToCenter(self.lander.GetWorldVector(f_v), True)
            elif action in [1, 3]: # Side Engines
                f_s = (1.0 if action==3 else -1.0) * (1.0 * 5.0) 
                self.lander.ApplyForceToCenter(self.lander.GetWorldVector((f_s, 0)), True)

        state = self._get_observation()
        
        # 3. Custom Reward (Distance to Pad)
        dist_x = abs(state[0])
        dist_y = abs(state[1])
        shaping = -100 * (dist_x + dist_y) - 100 * (abs(state[2]) + abs(state[3])) - 100 * abs(state[4])
        
        if self.custom_prev_shaping is not None:
            reward = shaping - self.custom_prev_shaping
        else:
            reward = 0
        self.custom_prev_shaping = shaping
        
        if action != 0:
            reward -= 0.1

        # 4. Success/Failure Detection
        terminated = False
        
        # Check if landed on the pad properly
        # state[6], state[7] are leg contacts
        legs_contact = (state[6] > 0 and state[7] > 0)
        on_pad = (dist_x < 0.05) # Centered on pad
        safe_vel = (abs(state[2]) < 0.1 and abs(state[3]) < 0.1) # Safe impact velocity
        safe_angle = (abs(state[4]) < 0.2) # Safe angle (~11 degrees)

        if self.game_over or (legs_contact and safe_vel):
            terminated = True
            if legs_contact and on_pad and safe_vel and safe_angle:
                self.mission_status = 'success'
                reward += 100
            else:
                self.mission_status = 'failed'
                reward -= 100
        
        # Custom Out of bounds check
        pos_x = self.lander.position.x
        pos_y = self.lander.position.y
        if pos_x < 0 or pos_x > WORLD_W or pos_y > WORLD_H * 2 or pos_y < 0:
            terminated = True
            self.mission_status = 'failed'
            reward = -100

        if self.fuel <= 0 and not terminated:
            # Check if it can still fall to ground
            if self.lander.linearVelocity.y > 0 and pos_y < 10:
                pass # Still falling
            elif self.lander.linearVelocity.y <= 0 and pos_y < 5:
                terminated = True
                self.mission_status = 'failed' # Out of fuel and stuck/crashed

        truncated = False
        info['mission_status'] = self.mission_status
        info['fuel'] = self.fuel
        
        return state, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode is None: return

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((CUSTOM_VIEWPORT_W, CUSTOM_VIEWPORT_H))
            pygame.display.set_caption("LunarLanderRL - NASA High Fidelity")
        
        if self.clock is None: self.clock = pygame.time.Clock()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True
            elif event.type == pygame.MOUSEWHEEL:
                self.zoom *= (1.1 if event.y > 0 else 0.9)
                self.zoom = max(0.1, min(self.zoom, 10.0))
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    self.mouse_dragging = True
                    self.last_mouse_pos = event.pos
                    self.camera_mode = 'manual'
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_dragging:
                    dx = (event.pos[0] - self.last_mouse_pos[0]) / self.zoom
                    dy = (event.pos[1] - self.last_mouse_pos[1]) / self.zoom
                    self.camera_x -= dx
                    self.camera_y -= dy
                    self.last_mouse_pos = event.pos
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    self.camera_mode = 'focus'
                if event.key == pygame.K_q:
                    self.game_over = True
                if event.key in [pygame.K_a, pygame.K_s, pygame.K_d, pygame.K_w]:
                    self.camera_mode = 'manual'

        # Keyboard Manual Camera
        keys = pygame.key.get_pressed()
        cam_speed = 15 / self.zoom
        if keys[pygame.K_a]: self.camera_x -= cam_speed
        if keys[pygame.K_d]: self.camera_x += cam_speed
        if keys[pygame.K_w]: self.camera_y -= cam_speed
        if keys[pygame.K_s]: self.camera_y += cam_speed

        # Focus Camera Smoothing
        if self.camera_mode == 'focus':
            target_x = self.lander.position.x * SCALE - CUSTOM_VIEWPORT_W // 2
            target_y = CUSTOM_VIEWPORT_H - self.lander.position.y * SCALE - CUSTOM_VIEWPORT_H // 2
            self.camera_x += (target_x - self.camera_x) * 0.1
            self.camera_y += (target_y - self.camera_y) * 0.1

        def to_screen(x, y):
            cx, cy = self.camera_x + CUSTOM_VIEWPORT_W // 2, self.camera_y + CUSTOM_VIEWPORT_H // 2
            screen_x = (x - cx) * self.zoom + CUSTOM_VIEWPORT_W // 2
            screen_y = (y - cy) * self.zoom + CUSTOM_VIEWPORT_H // 2
            return int(screen_x), int(screen_y)

        self.surf = pygame.Surface((CUSTOM_VIEWPORT_W, CUSTOM_VIEWPORT_H))
        self.surf.fill((2, 2, 8)) # Deep space
        
        # 1. Stars
        for star in self.stars:
            sx, sy = to_screen(star['pos'][0] - self.camera_x * 0.1, star['pos'][1] - self.camera_y * 0.1)
            if 0 <= sx <= CUSTOM_VIEWPORT_W and 0 <= sy <= CUSTOM_VIEWPORT_H:
                pygame.draw.circle(self.surf, star['color'], (sx, sy), max(1, int(star['size'] * self.zoom)))

        # 2. Lunar Body (Spherical Moon Core)
        mcx, mcy = to_screen(self.moon_center[0] * SCALE, CUSTOM_VIEWPORT_H - self.moon_center[1] * SCALE)
        moon_scr_radius = int(MOON_RADIUS * SCALE * self.zoom)
        if -moon_scr_radius < mcx < CUSTOM_VIEWPORT_W + moon_scr_radius:
            # Draw the main circular body
            pygame.draw.circle(self.surf, (20, 20, 22), (mcx, mcy), moon_scr_radius)
            # Subtle crater highlights on the core
            pygame.draw.circle(self.surf, (28, 28, 30), (mcx, mcy), moon_scr_radius, max(1, int(2 * self.zoom)))

        # 3. Lunar Surface Detail (Mountains/Terrain)
        for p in self.moon_polys:
            screen_poly = [to_screen(v[0] * SCALE, CUSTOM_VIEWPORT_H - v[1] * SCALE) for v in p]
            if any(-500 < x < CUSTOM_VIEWPORT_W + 500 for x, _ in screen_poly):
                pygame.draw.polygon(self.surf, (30, 30, 32), screen_poly)
                pygame.draw.line(self.surf, (110, 110, 115), screen_poly[0], screen_poly[1], max(1, int(1 * self.zoom)))

        # 4. Landing Pad
        ps1 = to_screen(self.helipad_x1 * SCALE, CUSTOM_VIEWPORT_H - self.helipad_y * SCALE)
        ps2 = to_screen(self.helipad_x2 * SCALE, CUSTOM_VIEWPORT_H - self.helipad_y * SCALE)
        # Pad is mostly flat but follows arc slightly - for now we just draw the line at the top
        pygame.draw.line(self.surf, (50, 255, 150), ps1, ps2, max(1, int(3 * self.zoom)))

        # 4. Exhaust Particles (High Fidelity)
        for obj in self.particles:
            ttl = getattr(obj, 'ttl', 0)
            if ttl <= 0: continue
            
            pos = obj.transform * (0, 0)
            px, py = to_screen(pos[0] * SCALE, CUSTOM_VIEWPORT_H - pos[1] * SCALE)
            
            # Use color1 to distinguish engine type if possible
            # Particles with higher color values are usually main engine
            is_main = False
            c1 = getattr(obj, 'color1', (0,0,0))
            if isinstance(c1, (list, tuple)) and len(c1) > 0:
                if c1[0] > 0.5: is_main = True

            if is_main:
                color = (int(200 * ttl), int(220 + 35 * ttl), 255)
                size = int((2.2 - ttl) * 7 * self.zoom)
            else:
                color = (255, int(180 + 75 * ttl), int(100 * ttl))
                size = int((1.8 - ttl) * 4 * self.zoom)

            if size > 0:
                # Outer glow
                glow_size = size * 3
                s = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*color, int(80 * ttl)), (glow_size, glow_size), glow_size)
                self.surf.blit(s, (px - glow_size, py - glow_size))
                # Core
                pygame.draw.circle(self.surf, (255, 255, 255, int(255 * ttl)), (px, py), max(1, size // 2))

        # 5. NASA Lander Rendering
        trans = self.lander.transform
        def get_screen_poly(poly):
            world_poly = [trans * (v[0]/SCALE, v[1]/SCALE) for v in poly]
            return [to_screen(v[0] * SCALE, CUSTOM_VIEWPORT_H - v[1] * SCALE) for v in world_poly]

        # Draw Descent Stage (Gold)
        pygame.draw.polygon(self.surf, (180, 150, 40), get_screen_poly(DESCENT_STAGE_POLY))
        pygame.draw.aalines(self.surf, (220, 190, 80), True, get_screen_poly(DESCENT_STAGE_POLY))

        # Draw Ascent Stage (Silver/Grey)
        pygame.draw.polygon(self.surf, (130, 135, 140), get_screen_poly(ASCENT_STAGE_POLY))
        pygame.draw.aalines(self.surf, (200, 205, 210), True, get_screen_poly(ASCENT_STAGE_POLY))

        # Draw Cockpit Window
        pygame.draw.polygon(self.surf, (30, 40, 60), get_screen_poly(COCKPIT_WINDOW_POLY))

        # Draw Legs
        for leg in self.legs:
            for f in leg.fixtures:
                l_trans = f.body.transform
                path = [l_trans * v for v in f.shape.vertices]
                screen_path = [to_screen(v[0] * SCALE, CUSTOM_VIEWPORT_H - v[1] * SCALE) for v in path]
                pygame.draw.polygon(self.surf, (90, 95, 100), screen_path)

        # 6. HUD and Overlays
        ui_font = pygame.font.SysFont('Consolas', 20, bold=True)
        
        # Fuel Bar
        fuel_rect = pygame.Rect(CUSTOM_VIEWPORT_W - 220, 20, 200, 25)
        pygame.draw.rect(self.surf, (40, 40, 45), fuel_rect, border_radius=5)
        # Fuel Color
        fuel_pct = self.fuel / 300.0
        if fuel_pct > 0.5: f_col = (0, 255, 100)
        elif fuel_pct > 0.2: f_col = (255, 200, 0)
        else: f_col = (255, 50, 50)
        
        pygame.draw.rect(self.surf, f_col, (fuel_rect.x, fuel_rect.y, int(fuel_rect.width * fuel_pct), fuel_rect.height), border_radius=5)
        txt_fuel = ui_font.render(f"FUEL: {int(self.fuel)}L", True, (255, 255, 255))
        self.surf.blit(txt_fuel, (fuel_rect.x + 50, fuel_rect.y + 2))

        # Mission Status Overlay
        if self.mission_status:
            overlay = pygame.Surface((CUSTOM_VIEWPORT_W, 100), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.surf.blit(overlay, (0, CUSTOM_VIEWPORT_H // 2 - 50))
            
            msg = "TOUCHDOWN CONFIRMED" if self.mission_status == 'success' else "VESSEL DESTROYED"
            col = (0, 255, 150) if self.mission_status == 'success' else (255, 50, 50)
            status_font = pygame.font.SysFont('Arial', 48, bold=True)
            txt_status = status_font.render(msg, True, col)
            self.surf.blit(txt_status, (CUSTOM_VIEWPORT_W // 2 - txt_status.get_width() // 2, CUSTOM_VIEWPORT_H // 2 - 30))

        # Controls Menu
        menu_items = ["F: Focus", "WASD or Drag: Move", "Scroll: Zoom", "Q: Quit"]
        for i, text in enumerate(menu_items):
            self.surf.blit(ui_font.render(text, True, (150, 150, 160)), (20, 20 + i * 25))

        if self.render_mode == "human":
            self.screen.blit(self.surf, (0, 0))
            self.clock.tick(FPS)
            pygame.display.flip()
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2))

        if self.render_mode == "human":
            self.screen.blit(self.surf, (0, 0))
            # pygame.event.pump() # Removed as we use event.get()
            self.clock.tick(FPS)
            pygame.display.flip()
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2))

if __name__ == "__main__":
    env = VastSpaceLander(render_mode="human")
    state, _ = env.reset()
    for _ in range(500):
        state, reward, done, _, _ = env.step(0)
        env.render()
        if done: break
    env.close()
