import pygame, random
import numpy as np
from .entity import Entity
from .render import blit

from hashlib import md5

from math import radians, cos, sin, sqrt

LEFT = 0
RIGHT = 1
DOWN = 2
UP = 3
LEFT_UP = 4
RIGHT_UP = 5
DOWN_RIGHT = 6
DOWN_LEFT = 7
# NOP = 8

ACTION_LOOKUP = {
    LEFT: [pygame.K_LEFT],
    RIGHT: [pygame.K_RIGHT],
    DOWN: [pygame.K_DOWN],
    UP: [pygame.K_UP],
    LEFT_UP: [pygame.K_LEFT, pygame.K_UP],
    RIGHT_UP: [pygame.K_RIGHT, pygame.K_UP],
    DOWN_RIGHT: [pygame.K_DOWN, pygame.K_RIGHT],
    DOWN_LEFT: [pygame.K_DOWN, pygame.K_LEFT]
}


class Environment:
    def __init__(self, scale=1, max_projectiles=60, render=False, keyboard=False, \
                 seed=random.randint(0, 2147483647), fov_size=50, actionResolver=None, max_age=None, \
                 border_dimensions=(100,100), framerate=25):
        self.dimensions = (fov_size * 2, fov_size * 2)

        self.framerate = framerate

        self.clear_raster()

        self.max_age = max_age

        self.background = None

        self.actionResolver = actionResolver

        self.max_projectiles = max_projectiles

        self.scale = scale

        self.total_reward = 0

        self.fov_size = fov_size
        self.fov_length = sqrt((2 * fov_size) ** 2)

        self.render = render
        self.keyboard = keyboard
        self.screen = None
        self.finished = False

        self.player = None

        self.border_dimensions = border_dimensions

        self._gen_player()
        self.left = False
        self.right = False
        self.up = False
        self.down = False

        self.projectiles = []

        self.n_collisions = 0

        self.age = 0

        self.fitness = 0

        random.seed(seed)

        self.spawn_segments = [
            ((0, -self.player.size), (self.border_dimensions[0] - self.player.size, -self.player.size), (0, 180)), \
            ((-self.player.size, 0), (-self.player.size, self.border_dimensions[1] - self.player.size), (-90, 90)), \
            ((0, self.border_dimensions[1] + self.player.size), \
             (self.border_dimensions[0] - self.player.size, self.border_dimensions[1] + self.player.size), (180, 360)), \
            ((self.border_dimensions[0] + self.player.size, 0), (self.border_dimensions[0] + self.player.size, \
                                                          self.border_dimensions[1] - self.player.size), (90, 180))]

        self.valid_keys = {pygame.K_LEFT: 'left', pygame.K_RIGHT: 'right', pygame.K_DOWN: 'down', pygame.K_UP: 'up'}

        self._initialize_render()

    def _handle_events(self):
        for event in pygame.event.get():
            self._handle_event(event)

    def _draw_rect(self, x, y, width, height):
        if self.render:
            pygame.draw.rect(self.background, (255, 255, 255),
                             (x - self.player.x + self.dimensions[0] / 2,
                              y - self.player.y + self.dimensions[1] / 2, width, height))
        else:
            blit(self.raster, (x - self.player.x + self.dimensions[0] / 2, y - self.player.y + self.dimensions[1] / 2, width, height))

    def _gen_player(self):
        self.player = Entity(x=self.border_dimensions[0] / 2, y=self.border_dimensions[1] / 2, size=self.scale)

    def _entities_nearby(self):
        for entity in self.projectiles:
            dx = self.player.x - entity.x
            dy = self.player.y - entity.y
            dist = sqrt(dx**2 + dy**2)
            if dist <= self.fov_length:
                return True
        return False

    def _render_boundaries(self):
        if self.player:
            self._draw_rect(0, -self.player.size, self.border_dimensions[0], self.player.size)
            self._draw_rect(0, self.border_dimensions[1], self.border_dimensions[0], self.player.size)
            self._draw_rect(-self.player.size, 0, self.player.size, self.border_dimensions[1])
            self._draw_rect(self.border_dimensions[0], 0, self.player.size, self.border_dimensions[1])

    def _initialize_render(self):
        if self.render:
            pygame.display.init()
            self.screen = pygame.display.set_mode(self.dimensions)

        self.background = pygame.Surface(self.dimensions)
        self.background.fill((255, 255, 255))

    @property
    def done(self):
        return self.age >= 5000

    @property
    def pygame_raster_array(self):
        arr = np.array(pygame.surfarray.array2d(self.background), dtype=np.float32)
        return np.reshape(arr / 16777215, self.dimensions)

    def reset_keys(self):
        for key in self.valid_keys.values():
            setattr(self, key, False)

    def _set_keys(self, keys, state):
        for key in keys:
            self._set_key(key, state)

    def _set_key(self, key, state):
        if key in self.valid_keys:
            setattr(self, self.valid_keys[key], state)

    def _handle_event(self, event):
        if event.type == pygame.QUIT:
            self.finished = True
        elif event.type == pygame.KEYDOWN:
            self._set_key(event.key, True)
        elif event.type == pygame.KEYUP:
            self._set_key(event.key, False)

    def _handle_player_movement(self):
        prev_x = self.player.x
        prev_y = self.player.y

        if self.left:
            self.player.x -= 1

        if self.right:
            self.player.x += 1

        if self.down:
            self.player.y += 1

        if self.up:
            self.player.y -= 1

        if self._out_of_bounds(self.player):
            self.player.x = prev_x
            self.player.y = prev_y
            return True

        for entity in self.projectiles:
            if entity.collides(self.player):
                return True

        return False


    def _out_of_bounds(self, entity):
        return entity.x > self.border_dimensions[0] - entity.size or entity.x <= 0 \
                or entity.y > self.border_dimensions[1] - entity.size or entity.y <= 0

    def get_raster(self):
        if self.render:
            return self.pygame_raster_array
        return self.raster

    @property
    def hash(self):
        m = md5()
        m.update(self.get_raster())
        return m.hexdigest()

    def step(self, action):
        if action in ACTION_LOOKUP:
            self._set_keys(ACTION_LOOKUP[action], True)
        collides = self._tick()
        self.reset_keys()

        reward = -1 if collides else 1

        if reward == -1:
            self.n_collisions += 1

        return self.get_raster(), reward # (state, reward)

    # Spawn an entity
    def _spawn_projectile(self):
        if len(self.projectiles) < self.max_projectiles:
            segment = random.choice(self.spawn_segments)
            pos = (random.uniform(segment[0][0], segment[1][0]), random.uniform(segment[0][1], segment[1][1]))
            angle = radians(random.randint(segment[2][0], segment[2][1]))
            self.projectiles.append(Entity(x=pos[0] - cos(angle)*10, y=pos[1]-sin(angle)*10, size=self.scale, vx=cos(angle), vy=sin(angle), max_age=self.fov_length + 15))

    def clear_raster(self):
        self.raster = np.zeros((self.dimensions[0], self.dimensions[1]))

    def _render_entity(self, entity):
        if self.player:
            if self.render:
                pygame.draw.rect(self.background, (255, 255, 255),
                             (entity.position[0] - self.player.x + self.dimensions[0] / 2,
                              entity.position[1] - self.player.y + self.dimensions[1] / 2, entity.size, entity.size))
            else:
                blit(self.raster, (entity.position[0] - self.player.x + self.dimensions[0] / 2,
                              entity.position[1] - self.player.y + self.dimensions[1] / 2, entity.size, entity.size))

    def _render_projectiles(self):
        for entity in self.projectiles:
            self._render_entity(entity)

    def _move_projectiles(self):
        to_remove = []

        for entity in self.projectiles:
            entity.update()

            if entity.should_delete:
                to_remove.append(entity)

        for entity in to_remove:
            self.projectiles.remove(entity)

    def _tick(self):
        self.background.fill((0, 0, 0))

        collides = False

        if self.player:
            collides = self._handle_player_movement()
            if self.render:
                pygame.draw.rect(self.background, (255, 255, 255), (self.dimensions[0] / 2, self.dimensions[1] / 2, \
                                                            self.player.size, self.player.size))
            else:
                blit(self.raster, (self.dimensions[0] / 2, self.dimensions[1] / 2, \
                                                                    self.player.size, self.player.size))

        self._move_projectiles()
        self._render_projectiles()

        self._render_boundaries()

        self._spawn_projectile()

        self.age += 1

        if collides:
            self.total_reward -= 1
        else:
            self.total_reward += 1

        return collides

    def run(self):
        while not self.finished:
            if self.actionResolver:
                self.step(self.actionResolver(self))

            if self.keyboard or self.render:
                self._handle_events()
            else:
                self.clear_raster()

            if self.render:
                self.screen.blit(self.background, (0, 0))

            if not self.actionResolver:
                self._tick()

            if self.render:
                pygame.display.flip()
                pygame.time.wait(self.framerate)

            if self.max_age is not None and self.age > self.max_age:
                self.finished = True
