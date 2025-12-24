"""
Mini Isaac - Gymnasium Environment for RL Training

A lightweight Isaac-style roguelike environment for reinforcement learning.
This is a pure Python implementation that mirrors the Rust game logic.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
import random
import math


class MiniIsaacEnv(gym.Env):
    """
    A Gymnasium environment for Mini Isaac roguelike game.

    Observation Space:
        - player_x, player_y: normalized position (0-1)
        - player_hp: current health (0-6)
        - enemy_positions: flattened array of 5 nearest enemies (x, y, hp)
        - nearest_enemy_dir: direction to nearest enemy (dx, dy)

    Action Space (Discrete 13):
        0: Noop
        1-8: Move (8 directions)
        9-12: Shoot (up, down, left, right)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # Constants (matching Rust game)
    TILE_SIZE = 32.0
    ROOM_WIDTH = 13
    ROOM_HEIGHT = 9
    SCREEN_WIDTH = TILE_SIZE * ROOM_WIDTH
    SCREEN_HEIGHT = TILE_SIZE * ROOM_HEIGHT + 64.0

    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 3000):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps

        # Observation space: player pos(2) + hp(1) + 5 enemies(15) + nearest dir(2) = 20
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(20,), dtype=np.float32
        )

        # Action space: 13 discrete actions
        self.action_space = spaces.Discrete(13)

        # Game state
        self.player = None
        self.enemies = []
        self.tears = []
        self.room_cleared = False
        self.step_count = 0
        self.score = 0

        # For rendering
        self.screen = None
        self.clock = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Initialize player
        self.player = {
            'x': self.SCREEN_WIDTH / 2,
            'y': (self.SCREEN_HEIGHT - 64) / 2 + 32,
            'vx': 0, 'vy': 0,
            'hp': 6, 'max_hp': 6,
            'damage': 1.0,
            'speed': 150.0,
            'fire_rate': 0.3,
            'fire_timer': 0.0,
            'invincible_timer': 0.0,
            'tears_speed': 300.0
        }

        # Spawn enemies
        self.enemies = []
        num_enemies = self.np_random.integers(3, 6)
        for _ in range(num_enemies):
            enemy_type = self.np_random.choice(['fly', 'gaper', 'shooter'])
            x = self.np_random.integers(2, self.ROOM_WIDTH - 2) * self.TILE_SIZE + self.TILE_SIZE / 2
            y = self.np_random.integers(2, self.ROOM_HEIGHT - 2) * self.TILE_SIZE + self.TILE_SIZE / 2 + 32

            hp_map = {'fly': 2.0, 'gaper': 4.0, 'shooter': 3.0}
            dmg_map = {'fly': 1.0, 'gaper': 1.0, 'shooter': 0.0}

            self.enemies.append({
                'x': x, 'y': y,
                'vx': 0, 'vy': 0,
                'hp': hp_map[enemy_type],
                'damage': dmg_map[enemy_type],
                'type': enemy_type,
                'timer': self.np_random.random() * 2,
                'shoot_timer': 0.0
            })

        self.tears = []
        self.room_cleared = False
        self.step_count = 0
        self.score = 0

        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        dt = 1.0 / 60.0
        self.step_count += 1

        old_hp = self.player['hp']
        old_enemies = len(self.enemies)

        # Apply action
        self._apply_action(action, dt)

        # Update game state
        self._update_player(dt)
        self._update_enemies(dt)
        self._update_tears(dt)
        self._check_collisions()

        # Calculate reward
        reward = 0.0

        # Reward for killing enemies
        enemies_killed = old_enemies - len(self.enemies)
        reward += enemies_killed * 10.0

        # Penalty for taking damage
        damage_taken = old_hp - self.player['hp']
        reward -= damage_taken * 5.0

        # Small survival reward
        reward += 0.01

        # Check termination
        terminated = False
        truncated = False

        if self.player['hp'] <= 0:
            terminated = True
            reward -= 50.0

        if len(self.enemies) == 0 and not self.room_cleared:
            self.room_cleared = True
            reward += 50.0
            terminated = True  # Room cleared = episode done

        if self.step_count >= self.max_steps:
            truncated = True

        info = {
            'score': self.score,
            'enemies_remaining': len(self.enemies),
            'player_hp': self.player['hp']
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(20, dtype=np.float32)

        # Player position (normalized)
        obs[0] = self.player['x'] / self.SCREEN_WIDTH
        obs[1] = self.player['y'] / self.SCREEN_HEIGHT
        obs[2] = self.player['hp'] / self.player['max_hp']

        # Sort enemies by distance
        sorted_enemies = sorted(
            self.enemies,
            key=lambda e: (e['x'] - self.player['x'])**2 + (e['y'] - self.player['y'])**2
        )

        # 5 nearest enemies (x, y, hp normalized)
        for i, enemy in enumerate(sorted_enemies[:5]):
            idx = 3 + i * 3
            obs[idx] = enemy['x'] / self.SCREEN_WIDTH
            obs[idx + 1] = enemy['y'] / self.SCREEN_HEIGHT
            obs[idx + 2] = enemy['hp'] / 4.0  # Max enemy hp is ~4

        # Nearest enemy direction
        if sorted_enemies:
            dx = sorted_enemies[0]['x'] - self.player['x']
            dy = sorted_enemies[0]['y'] - self.player['y']
            dist = max(1.0, math.sqrt(dx*dx + dy*dy))
            obs[18] = dx / dist
            obs[19] = dy / dist

        return obs

    def _apply_action(self, action: int, dt: float):
        speed = self.player['speed']

        # Reset velocity
        self.player['vx'] = 0
        self.player['vy'] = 0

        if action == 0:  # Noop
            pass
        elif action == 1:  # Up
            self.player['vy'] = -speed
        elif action == 2:  # Down
            self.player['vy'] = speed
        elif action == 3:  # Left
            self.player['vx'] = -speed
        elif action == 4:  # Right
            self.player['vx'] = speed
        elif action == 5:  # Up-Left
            self.player['vx'] = -speed * 0.707
            self.player['vy'] = -speed * 0.707
        elif action == 6:  # Up-Right
            self.player['vx'] = speed * 0.707
            self.player['vy'] = -speed * 0.707
        elif action == 7:  # Down-Left
            self.player['vx'] = -speed * 0.707
            self.player['vy'] = speed * 0.707
        elif action == 8:  # Down-Right
            self.player['vx'] = speed * 0.707
            self.player['vy'] = speed * 0.707
        elif action == 9:  # Shoot Up
            self._try_shoot(0, -1)
        elif action == 10:  # Shoot Down
            self._try_shoot(0, 1)
        elif action == 11:  # Shoot Left
            self._try_shoot(-1, 0)
        elif action == 12:  # Shoot Right
            self._try_shoot(1, 0)

    def _try_shoot(self, dx: float, dy: float):
        if self.player['fire_timer'] <= 0:
            speed = self.player['tears_speed']
            self.tears.append({
                'x': self.player['x'],
                'y': self.player['y'],
                'vx': dx * speed,
                'vy': dy * speed,
                'damage': self.player['damage'],
                'from_player': True,
                'lifetime': 1.0
            })
            self.player['fire_timer'] = self.player['fire_rate']

    def _update_player(self, dt: float):
        # Update timers
        self.player['fire_timer'] -= dt
        self.player['invincible_timer'] -= dt

        # Update position with collision
        new_x = self.player['x'] + self.player['vx'] * dt
        new_y = self.player['y'] + self.player['vy'] * dt

        margin = 12.0
        new_x = max(margin, min(self.SCREEN_WIDTH - margin, new_x))
        new_y = max(32 + margin, min(self.SCREEN_HEIGHT - margin, new_y))

        self.player['x'] = new_x
        self.player['y'] = new_y

    def _update_enemies(self, dt: float):
        player_x = self.player['x']
        player_y = self.player['y']

        for enemy in self.enemies:
            enemy['timer'] += dt
            enemy['shoot_timer'] += dt

            if enemy['type'] == 'fly':
                # Random movement
                if enemy['timer'] > 0.5:
                    enemy['vx'] = self.np_random.uniform(-80, 80)
                    enemy['vy'] = self.np_random.uniform(-80, 80)
                    enemy['timer'] = 0

            elif enemy['type'] == 'gaper':
                # Chase player
                dx = player_x - enemy['x']
                dy = player_y - enemy['y']
                dist = max(1.0, math.sqrt(dx*dx + dy*dy))
                enemy['vx'] = dx / dist * 60
                enemy['vy'] = dy / dist * 60

            elif enemy['type'] == 'shooter':
                # Shoot at player
                if enemy['shoot_timer'] > 2.0:
                    dx = player_x - enemy['x']
                    dy = player_y - enemy['y']
                    dist = max(1.0, math.sqrt(dx*dx + dy*dy))
                    self.tears.append({
                        'x': enemy['x'],
                        'y': enemy['y'],
                        'vx': dx / dist * 150,
                        'vy': dy / dist * 150,
                        'damage': 1.0,
                        'from_player': False,
                        'lifetime': 2.0
                    })
                    enemy['shoot_timer'] = 0

            # Update position
            enemy['x'] += enemy['vx'] * dt
            enemy['y'] += enemy['vy'] * dt

            # Clamp to room
            enemy['x'] = max(48, min(self.SCREEN_WIDTH - 48, enemy['x']))
            enemy['y'] = max(80, min(self.SCREEN_HEIGHT - 48, enemy['y']))

    def _update_tears(self, dt: float):
        for tear in self.tears:
            tear['x'] += tear['vx'] * dt
            tear['y'] += tear['vy'] * dt
            tear['lifetime'] -= dt

        # Remove dead tears
        self.tears = [t for t in self.tears if t['lifetime'] > 0
                      and 0 < t['x'] < self.SCREEN_WIDTH
                      and 32 < t['y'] < self.SCREEN_HEIGHT]

    def _check_collisions(self):
        player_rect = (
            self.player['x'] - 12, self.player['y'] - 12,
            self.player['x'] + 12, self.player['y'] + 12
        )

        # Player tears vs enemies
        for tear in self.tears:
            if not tear['from_player']:
                continue
            tear_rect = (tear['x'] - 6, tear['y'] - 6, tear['x'] + 6, tear['y'] + 6)
            for enemy in self.enemies:
                enemy_rect = (enemy['x'] - 10, enemy['y'] - 10, enemy['x'] + 10, enemy['y'] + 10)
                if self._rects_overlap(tear_rect, enemy_rect):
                    enemy['hp'] -= tear['damage']
                    tear['lifetime'] = 0
                    break

        # Remove dead enemies
        killed = [e for e in self.enemies if e['hp'] <= 0]
        self.score += len(killed) * 10
        self.enemies = [e for e in self.enemies if e['hp'] > 0]

        # Enemy tears vs player
        if self.player['invincible_timer'] <= 0:
            for tear in self.tears:
                if tear['from_player']:
                    continue
                tear_rect = (tear['x'] - 6, tear['y'] - 6, tear['x'] + 6, tear['y'] + 6)
                if self._rects_overlap(tear_rect, player_rect):
                    self.player['hp'] -= 1
                    self.player['invincible_timer'] = 1.0
                    tear['lifetime'] = 0
                    break

        # Contact damage
        if self.player['invincible_timer'] <= 0:
            for enemy in self.enemies:
                if enemy['damage'] > 0:
                    enemy_rect = (enemy['x'] - 10, enemy['y'] - 10, enemy['x'] + 10, enemy['y'] + 10)
                    if self._rects_overlap(player_rect, enemy_rect):
                        self.player['hp'] -= int(enemy['damage'])
                        self.player['invincible_timer'] = 1.0
                        break

    @staticmethod
    def _rects_overlap(r1, r2):
        return not (r1[2] < r2[0] or r1[0] > r2[2] or r1[3] < r2[1] or r1[1] > r2[3])

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_human()

    def _render_frame(self) -> np.ndarray:
        """Render to numpy array for video recording."""
        try:
            import pygame
        except ImportError:
            return np.zeros((int(self.SCREEN_HEIGHT), int(self.SCREEN_WIDTH), 3), dtype=np.uint8)

        if self.screen is None:
            pygame.init()
            self.screen = pygame.Surface((int(self.SCREEN_WIDTH), int(self.SCREEN_HEIGHT)))

        self._draw_game(self.screen)
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def _render_human(self):
        """Render to screen for human viewing."""
        try:
            import pygame
        except ImportError:
            print("pygame not installed, cannot render")
            return

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((int(self.SCREEN_WIDTH), int(self.SCREEN_HEIGHT)))
            self.clock = pygame.time.Clock()

        self._draw_game(self.screen)
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _draw_game(self, surface):
        """Draw game state to a pygame surface."""
        try:
            import pygame
        except ImportError:
            return

        # Background
        surface.fill((40, 40, 50))

        # Draw floor
        for y in range(self.ROOM_HEIGHT):
            for x in range(self.ROOM_WIDTH):
                px = x * self.TILE_SIZE
                py = y * self.TILE_SIZE + 32
                pygame.draw.rect(surface, (60, 50, 45), (px, py, self.TILE_SIZE, self.TILE_SIZE))

        # Draw tears
        for tear in self.tears:
            color = (100, 180, 255) if tear['from_player'] else (255, 100, 100)
            pygame.draw.circle(surface, color, (int(tear['x']), int(tear['y'])), 6)

        # Draw enemies
        for enemy in self.enemies:
            if enemy['type'] == 'fly':
                color, size = (100, 100, 100), 10
            elif enemy['type'] == 'gaper':
                color, size = (200, 100, 100), 14
            else:
                color, size = (100, 200, 100), 12
            pygame.draw.circle(surface, color, (int(enemy['x']), int(enemy['y'])), size)

        # Draw player
        color = (255, 220, 180)
        if self.player['invincible_timer'] > 0:
            color = (255, 255, 255) if (self.step_count % 10) < 5 else (200, 200, 200)
        pygame.draw.circle(surface, color, (int(self.player['x']), int(self.player['y'])), 12)

        # Eyes
        pygame.draw.circle(surface, (255, 255, 255), (int(self.player['x']) - 4, int(self.player['y']) - 2), 3)
        pygame.draw.circle(surface, (255, 255, 255), (int(self.player['x']) + 4, int(self.player['y']) - 2), 3)
        pygame.draw.circle(surface, (0, 0, 0), (int(self.player['x']) - 4, int(self.player['y']) - 2), 1)
        pygame.draw.circle(surface, (0, 0, 0), (int(self.player['x']) + 4, int(self.player['y']) - 2), 1)

    def close(self):
        if self.screen is not None:
            try:
                import pygame
                pygame.quit()
            except:
                pass
            self.screen = None


# Register the environment
gym.register(
    id='MiniIsaac-v0',
    entry_point='env:MiniIsaacEnv',
    max_episode_steps=3000,
)


if __name__ == "__main__":
    # Quick test
    env = MiniIsaacEnv(render_mode=None)
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    total_reward = 0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"Test episode reward: {total_reward:.2f}")
    print(f"Final info: {info}")
    env.close()
