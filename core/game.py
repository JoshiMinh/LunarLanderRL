import numpy as np
import gymnasium as gym
import Box2D
from Box2D.b2 import fixtureDef, polygonShape, revoluteJointDef
from gymnasium.envs.box2d.lunar_lander import LunarLander, INITIAL_RANDOM, SCALE, FPS

from core.constants import WORLD_W, WORLD_H, MODERN_LANDER_POLY
from core.terrain import generate_cosmos, build_lunar_surface
from core.renderer import Renderer

class VastSpaceLander(LunarLander):
    """
    Ultra-Realistic Scrolling Lunar Descent Simulation.
    Features a modern 'Starship' style lander and a vast surface.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fuel = 300.0 # Tripled fuel
        self.mission_status = None # 'success' or 'failed'
        self.user_quit = False
        self.user_skip = False
        self.step_count = 0
        self.max_episode_steps = 1500 # Reduced from 15,000 to prevent drifting episodes
        self.success_wait_steps = FPS * 10 if self.render_mode == "human" else int(FPS * 0.4)
        self.success_timer_steps = 0
        
        import core.constants as const
        self.stars = generate_cosmos(const.CUSTOM_VIEWPORT_W, const.CUSTOM_VIEWPORT_H)
        self.my_renderer = Renderer(self.render_mode)

        # Update Observation Space for fuel (9 elements)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.custom_prev_shaping = None
        self.user_skip = False
        self.step_count = 0
        self.success_timer_steps = 0
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

        surface_data = build_lunar_surface(self.world, self.np_random)
        self.moon = surface_data["moon"]
        self.moon_polys = surface_data["moon_polys"]
        self.moon_center = surface_data["moon_center"]
        self.helipad_y = surface_data["helipad_y"]
        self.helipad_x1 = surface_data["helipad_x1"]
        self.helipad_x2 = surface_data["helipad_x2"]

        # Create Modern Lander at high altitude
        H = WORLD_H
        W = WORLD_W
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
        import math
        normalized_angle = (self.lander.angle + math.pi) % (2 * math.pi) - math.pi
        
        state = [
            (pos.x - pad_center) / (WORLD_W / 2),
            (pos.y - (self.helipad_y + 20/SCALE)) / (WORLD_H / 2),
            vel.x * 0.2, # Standard LunarLander scale
            vel.y * 0.2, 
            normalized_angle,
            self.lander.angularVelocity * 0.05,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.fuel / 300.0, # 9th element: Fuel awareness
        ]
        return np.array(state, dtype=np.float32)

    def step(self, action):
        self.step_count += 1

        # User can mark the episode as failed manually.
        if self.user_skip:
            self.user_skip = False
            self.mission_status = 'failed'
            state = self._get_observation()
            info = {'mission_status': self.mission_status, 'fuel': self.fuel}
            return state, -100.0, True, False, info

        # During success showcase window, keep engines off and count down.
        if self.mission_status == 'success' and self.success_timer_steps > 0:
            action = 0

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
        
        # Explicit particle cleanup
        for obj in self.particles:
            if hasattr(obj, 'ttl'):
                obj.ttl -= 0.1 
        
        # 2. Boost Engine Power
        if self.fuel > 0:
            if action == 2: # Main Engine
                f_v = (0, 12.0 * 20.0) 
                self.lander.ApplyForceToCenter(self.lander.GetWorldVector(f_v), True)
            elif action in [1, 3]: # Side Engines
                f_s = (1.0 if action==3 else -1.0) * (2.0 * 20.0) 
                self.lander.ApplyForceToCenter(self.lander.GetWorldVector((f_s, 0)), True)

        state = self._get_observation()
        
        # 3. Custom Reward (Distance to Pad)
        dist_x = abs(state[0])
        dist_y = abs(state[1])
        # Reduced velocity penalty so the agent isn't terrified of falling
        shaping = -100 * (dist_x + dist_y) - 50 * (abs(state[2]) + abs(state[3])) - 100 * abs(state[4])
        
        if self.custom_prev_shaping is not None:
            reward = shaping - self.custom_prev_shaping
        else:
            reward = 0
        self.custom_prev_shaping = shaping

        # Small living cost to avoid policy stalling/hovering for too long.
        reward -= 0.03
        
        if action != 0:
            reward -= 0.1

        # Penalize sustained hover near the pad without committing to touchdown.
        no_leg_contact = (state[6] == 0.0 and state[7] == 0.0)
        near_pad = (dist_x < 0.15 and dist_y < 0.35)
        near_zero_vertical = abs(state[3]) < 0.05
        if self.mission_status != 'success' and no_leg_contact and near_pad and near_zero_vertical:
            reward -= 0.25

        # 4. Success/Failure Detection
        terminated = False
        
        # Check if landed on the pad properly
        legs_contact = (state[6] > 0 and state[7] > 0)
        on_pad = (dist_x < 0.05) # Centered on pad
        safe_vel = (abs(state[2]) < 0.1 and abs(state[3]) < 0.1) # Safe impact velocity
        safe_angle = (abs(state[4]) < 0.2) # Safe angle (~11 degrees)

        if self.mission_status == 'success':
            self.success_timer_steps -= 1
            reward = 0.0
            if self.success_timer_steps <= 0:
                terminated = True
        # Latch success and keep episode alive for a short showcase window.
        elif legs_contact and on_pad and safe_vel and safe_angle:
            self.mission_status = 'success'
            reward += 180
            self.success_timer_steps = self.success_wait_steps
        # Fail only on crash contact.
        elif self.game_over:
            terminated = True
            self.mission_status = 'failed'
            reward -= 120

        truncated = (self.step_count >= self.max_episode_steps) and (not terminated)
        if truncated and self.mission_status is None:
            self.mission_status = 'timeout'
        info['mission_status'] = self.mission_status
        info['fuel'] = self.fuel
        
        return state, float(reward), terminated, truncated, info

    def render(self):
        return self.my_renderer.render(self)

    def close(self):
        super().close()
        import pygame
        pygame.quit()
