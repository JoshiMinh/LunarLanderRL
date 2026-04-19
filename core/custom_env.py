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
MODERN_LANDER_POLY = [
    (-10, +25), (+10, +25), (+12, -20), (+15, -25), (-15, -25), (-12, -20)
]

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
        self.generate_cosmos()

    def generate_cosmos(self):
        """Generate distant realistic stars."""
        for _ in range(500):
            x = np.random.randint(0, CUSTOM_VIEWPORT_W * 4) # Wide star field for parallax
            y = np.random.randint(0, CUSTOM_VIEWPORT_H * 4)
            brightness = np.random.randint(100, 255)
            self.stars.append({'pos': [x, y], 'size': np.random.uniform(0.5, 1.5), 'color': (brightness, brightness, brightness)})

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        self._rebuild_world_for_realism()
        if self.render_mode == "human":
            self.render()
        return self._get_observation(), {}

    def _rebuild_world_for_realism(self):
        """Build a vast, rugged lunar surface."""
        self._destroy()
        self.world = Box2D.b2World(gravity=(0, self.gravity))
        from gymnasium.envs.box2d.lunar_lander import ContactDetector
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

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
        self.helipad_y = height[pad_idx] # Pad matches local height
        
        for i in range(pad_idx - 3, pad_idx + 4):
            height[i] = self.helipad_y

        # Smooth terrain but keep it rugged
        smooth_y = [
            0.5 * (height[i-1] + height[i]) if i > 0 else height[i]
            for i in range(CHUNKS)
        ]

        self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i+1], smooth_y[i+1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.5)
            self.sky_polys.append([p1, p2, (p2[0], H*2), (p1[0], H*2)])

        # Create Modern Lander at high altitude
        initial_y = H * 0.85
        initial_x = W / 2 + self.np_random.uniform(-W/10, W/10)
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in MODERN_LANDER_POLY]),
                density=5.0, friction=0.1, categoryBits=0x0010, maskBits=0x001, restitution=0.0,
            ),
        )
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
            vel.x * (WORLD_W / 2) / FPS,
            vel.y * (WORLD_H / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        return np.array(state, dtype=np.float32)

    def step(self, action):
        _, reward, terminated, truncated, info = super().step(action)
        state = self._get_observation()
        
        # Custom realistic distance-based reward
        dist_x = abs(state[0])
        dist_y = abs(state[1])
        shaping = -100 * (dist_x + dist_y) - 100 * (abs(state[2]) + abs(state[3])) - 100 * abs(state[4])
        
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping
        
        # Out of bounds check
        if pos_x := self.lander.position.x < 0 or self.lander.position.x > WORLD_W:
            terminated = True
            reward = -100

        return state, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None: return

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((CUSTOM_VIEWPORT_W, CUSTOM_VIEWPORT_H))
            pygame.display.set_caption("Lunar Descent - Realistic Simulation")
        
        if self.clock is None: self.clock = pygame.time.Clock()

        # Update Camera - Smooth follow with interpolation
        target_x = self.lander.position.x * SCALE - CUSTOM_VIEWPORT_W // 2
        target_y = CUSTOM_VIEWPORT_H - self.lander.position.y * SCALE - CUSTOM_VIEWPORT_H // 2
        self.camera_x += (target_x - self.camera_x) * 0.1
        self.camera_y += (target_y - self.camera_y) * 0.1

        self.surf = pygame.Surface((CUSTOM_VIEWPORT_W, CUSTOM_VIEWPORT_H))
        self.surf.fill((0, 0, 0)) # Absolute space black
        
        # 1. Draw Parallax Stars
        for star in self.stars:
            sx = (star['pos'][0] - self.camera_x * 0.2) % (CUSTOM_VIEWPORT_W * 2)
            sy = (star['pos'][1] - self.camera_y * 0.2) % (CUSTOM_VIEWPORT_H * 2)
            pygame.draw.circle(self.surf, star['color'], (int(sx), int(sy)), star['size'])

        # 2. Draw Terrain (Realistic Grayscale)
        for p in self.sky_polys:
            # Shift terrain by camera
            screen_poly = [
                (v[0] * SCALE - self.camera_x, CUSTOM_VIEWPORT_H - v[1] * SCALE - self.camera_y) 
                for v in p
            ]
            # Optimization: Only draw if visible
            if any(0 <= x <= CUSTOM_VIEWPORT_W for x, y in screen_poly):
                pygame.draw.polygon(self.surf, (20, 20, 22), screen_poly)
                pygame.draw.line(self.surf, (100, 100, 105), screen_poly[0], screen_poly[1], 2)

        # 3. Draw Prepared Landing Pad (Glowing accents)
        pad_x1 = self.helipad_x1 * SCALE - self.camera_x
        pad_x2 = self.helipad_x2 * SCALE - self.camera_x
        pad_y = CUSTOM_VIEWPORT_H - self.helipad_y * SCALE - self.camera_y
        if -200 < pad_x1 < CUSTOM_VIEWPORT_W + 200:
            pygame.draw.rect(self.surf, (40, 40, 45), (pad_x1, pad_y, pad_x2 - pad_x1, 5))
            # Glowing edges
            glow_time = (pygame.time.get_ticks() // 500) % 2
            blink_color = (0, 255, 150) if glow_time else (0, 100, 60)
            pygame.draw.circle(self.surf, blink_color, (int(pad_x1), int(pad_y)), 4)
            pygame.draw.circle(self.surf, blink_color, (int(pad_x2), int(pad_y)), 4)

        # 4. Draw Modern Lander (Starship Style)
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                screen_path = [
                    (v[0] * SCALE - self.camera_x, CUSTOM_VIEWPORT_H - v[1] * SCALE - self.camera_y) 
                    for v in path
                ]
                if obj == self.lander:
                    # Polished brushed metal look
                    pygame.draw.polygon(self.surf, (180, 185, 190), screen_path)
                    pygame.draw.aalines(self.surf, (220, 230, 240), True, screen_path)
                    # Heat shield (bottom)
                    if len(screen_path) >= 5:
                        pygame.draw.line(self.surf, (30, 30, 35), screen_path[3], screen_path[4], 3)
                else:
                    pygame.draw.polygon(self.surf, (60, 63, 65), screen_path)

        # 5. Thrust Particles (Realistic Blue-White Plasma)
        for obj in self.particles:
            obj.ttl -= 0.1
            if obj.ttl <= 0: continue
            pos = obj.transform * (0, 0)
            px = pos[0] * SCALE - self.camera_x
            py = CUSTOM_VIEWPORT_H - pos[1] * SCALE - self.camera_y
            alpha = int(max(0, obj.ttl) * 255)
            # Blue core with white outer
            size = int((1.5 - obj.ttl) * 4)
            pygame.draw.circle(self.surf, (100, 200, 255, alpha//2), (int(px), int(py)), size * 2)
            pygame.draw.circle(self.surf, (255, 255, 255, alpha), (int(px), int(py)), size)

        if self.render_mode == "human":
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
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
