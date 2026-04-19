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

# Vast Space Constants
CUSTOM_VIEWPORT_W = 1200
CUSTOM_VIEWPORT_H = 800

class VastSpaceLander(LunarLander):
    """
    HD Sci-Fi version of LunarLander with a much larger world and premium visuals.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stars = []
        self.nebulae = []
        self.generate_cosmos()

    def generate_cosmos(self):
        """Generate static background elements."""
        for _ in range(200):
            x = np.random.randint(0, CUSTOM_VIEWPORT_W)
            y = np.random.randint(0, CUSTOM_VIEWPORT_H)
            size = np.random.uniform(0.5, 2.0)
            brightness = np.random.randint(150, 255)
            self.stars.append({'pos': (x, y), 'size': size, 'color': (brightness, brightness, brightness)})
        
        colors = [(40, 0, 80, 40), (0, 40, 80, 40), (80, 0, 40, 40)]
        for _ in range(5):
            x = np.random.randint(0, CUSTOM_VIEWPORT_W)
            y = np.random.randint(0, CUSTOM_VIEWPORT_H)
            radius = np.random.randint(100, 300)
            color = colors[np.random.randint(0, len(colors))]
            self.nebulae.append({'pos': (x, y), 'radius': radius, 'color': color})

    def reset(self, *, seed=None, options=None):
        # Initialize seed for reproducibility
        gym.Env.reset(self, seed=seed)
        
        self._rebuild_world_for_vast_space()
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_observation(), {}

    def _rebuild_world_for_vast_space(self):
        """Re-initialize world objects with new dimensions."""
        self._destroy()
        self.world = Box2D.b2World(gravity=(0, self.gravity))
        from gymnasium.envs.box2d.lunar_lander import ContactDetector
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = CUSTOM_VIEWPORT_W / SCALE
        H = CUSTOM_VIEWPORT_H / SCALE

        # Create Terrain - More chunks for vaster space
        CHUNKS = 25
        height = self.np_random.uniform(0, H / 3, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        
        # Helipad in the middle
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 6 # Lower relative to height
        
        for i in range(CHUNKS // 2 - 2, CHUNKS // 2 + 3):
            height[i] = self.helipad_y

        smooth_y = [
            0.33 * (height[i - 1] + height[i + 0] + height[i + 1]) if 0 < i < CHUNKS else height[i]
            for i in range(CHUNKS)
        ]

        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)])
        )
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            # Sky polys should reach the top of the NEW viewport
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        # Create Lander at the top
        initial_y = H * 0.9
        initial_x = W / 2
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
                ),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,
                restitution=0.0,
            ),
        )
        self.lander.color1 = (100, 100, 150)
        self.lander.color2 = (50, 50, 80)

        self.lander.ApplyForceToCenter(
            (
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            ),
            True,
        )

        # Legs
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (100, 100, 150)
            leg.color2 = (50, 50, 80)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,
            )
            if i == -1:
                rjd.lowerAngle = +0.4
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.4
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

    def _get_observation(self):
        """Standard observation but with custom viewport scaling."""
        pos = self.lander.position
        vel = self.lander.linearVelocity
        W = CUSTOM_VIEWPORT_W / SCALE
        H = CUSTOM_VIEWPORT_H / SCALE
        
        state = [
            (pos.x - W / 2) / (W / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (H / 2),
            vel.x * (W / 2) / FPS,
            vel.y * (H / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        return np.array(state, dtype=np.float32)

    def step(self, action):
        _, reward, terminated, truncated, info = super().step(action)
        state = self._get_observation()
        shaping = (
            -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping
        if abs(state[0]) >= 1.0:
            terminated = True
            reward = -100
        return state, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((CUSTOM_VIEWPORT_W, CUSTOM_VIEWPORT_H))
            pygame.display.set_caption("Lunar Lander RL - HD Sci-Fi Edition")
        
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((CUSTOM_VIEWPORT_W, CUSTOM_VIEWPORT_H))
        self.surf.fill((5, 5, 20)) # Very dark blue
        
        # Draw subtle tech grid
        for x in range(0, CUSTOM_VIEWPORT_W, 100):
            pygame.draw.line(self.surf, (15, 25, 40), (x, 0), (x, CUSTOM_VIEWPORT_H))
        for y in range(0, CUSTOM_VIEWPORT_H, 100):
            pygame.draw.line(self.surf, (15, 25, 40), (0, y), (CUSTOM_VIEWPORT_W, y))
        
        # Draw Nebulae (soft glow)
        for neb in self.nebulae:
            neb_surf = pygame.Surface((neb['radius']*2, neb['radius']*2), pygame.SRCALPHA)
            for r in range(neb['radius'], 0, -20):
                pygame.draw.circle(neb_surf, list(neb['color'][:3]) + [int(40 * (1 - r/neb['radius']))], (neb['radius'], neb['radius']), r)
            self.surf.blit(neb_surf, (neb['pos'][0] - neb['radius'], neb['pos'][1] - neb['radius']))

        # Draw Stars
        for star in self.stars:
            pygame.draw.circle(self.surf, star['color'], star['pos'], star['size'])

        # 2. Draw Terrain (Sci-Fi Black/Neon)
        for p in self.sky_polys:
            scaled_poly = [(coord[0] * SCALE, CUSTOM_VIEWPORT_H - coord[1] * SCALE) for coord in p]
            pygame.draw.polygon(self.surf, (15, 15, 20), scaled_poly)
            # Draw surface line (neon blue pulse effect)
            pygame.draw.line(self.surf, (0, 100, 255), scaled_poly[0], scaled_poly[1], 3)
            pygame.draw.line(self.surf, (200, 230, 255), scaled_poly[0], scaled_poly[1], 1)

        # 3. Draw Landing Pad Flags (Neon)
        for x in [self.helipad_x1, self.helipad_x2]:
            px = x * SCALE
            py = CUSTOM_VIEWPORT_H - self.helipad_y * SCALE
            pygame.draw.line(self.surf, (255, 255, 255), (px, py), (px, py - 40), 2)
            pygame.draw.polygon(self.surf, (0, 255, 100), [(px, py-40), (px, py-30), (px+15, py-35)])

        # 4. Draw Particles (Plasma)
        for obj in self.particles:
            obj.ttl -= 0.1
            trans = obj.transform
            pos = trans * (0, 0)
            screen_x = pos[0] * SCALE
            screen_y = CUSTOM_VIEWPORT_H - pos[1] * SCALE
            alpha = int(max(0, obj.ttl) * 255)
            r, g, b = (50, 150, 255) if obj.ttl > 0.5 else (200, 200, 255)
            radius = int(max(1, (1.2 - obj.ttl) * 5))
            s = pygame.Surface((radius*4, radius*4), pygame.SRCALPHA)
            pygame.draw.circle(s, (r, g, b, alpha//2), (radius*2, radius*2), radius*2)
            pygame.draw.circle(s, (255, 255, 255, alpha), (radius*2, radius*2), radius)
            self.surf.blit(s, (screen_x - radius*2, screen_y - radius*2))

        # 5. Draw Lander (Detailed)
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                screen_path = [(v[0] * SCALE, CUSTOM_VIEWPORT_H - v[1] * SCALE) for v in path]
                if obj == self.lander:
                    pygame.draw.polygon(self.surf, (40, 45, 60), screen_path)
                    pygame.draw.aalines(self.surf, (100, 110, 130), True, screen_path)
                    if len(screen_path) >= 4:
                        pygame.draw.line(self.surf, (60, 70, 90), screen_path[0], screen_path[3], 1)
                    cockpit_center = (screen_path[0][0] + screen_path[5][0])/2, (screen_path[0][1] + screen_path[5][1])/2
                    pygame.draw.circle(self.surf, (0, 200, 255), (int(cockpit_center[0]), int(cockpit_center[1])), 4)
                else:
                    pygame.draw.polygon(self.surf, (70, 70, 80), screen_path)
                    pygame.draw.aalines(self.surf, (150, 150, 160), True, screen_path)

        if self.render_mode == "human":
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(FPS)
            pygame.display.flip()
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )

if __name__ == "__main__":
    env = VastSpaceLander(render_mode="human")
    state, _ = env.reset()
    for _ in range(500):
        state, reward, done, _, _ = env.step(0)
        env.render()
        if done: break
    env.close()
