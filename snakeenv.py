# Adapted from: https://github.com/TheAILearner/Snake-Game-using-OpenCV-Python/blob/master/snake_game_using_opencv.ipynb
# Get from Sentdex: https://www.youtube.com/watch?v=uKnjGn8fF70&list=PLQVvvaa0QuDf0O2DWwLZBfJeYY-JOeZB1&index=3
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random
import time
from collections import deque
import math
from enum import Enum

SNAKE_LEN_GOAL = 30


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
    score += 1
    return apple_position, score


def collision_with_boundaries(snake_head):
    if (
        snake_head[0] >= 500
        or snake_head[0] < 0
        or snake_head[1] >= 500
        or snake_head[1] < 0
    ):
        return 1
    else:
        return 0


def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0


def moveSnake(self, action):
    clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
    idx = clock_wise.index(self.direction)

    # Seguir recto.
    if action == 0:
        new_dir = clock_wise[idx]
    # Girar derecha.
    elif action == 1:
        next_idx = (idx + 1) % 4
        new_dir = clock_wise[next_idx]
    # Girar izquierda.
    else:
        next_idx = (idx - 1) % 4
        new_dir = clock_wise[next_idx]

    self.direction = new_dir

    x = self.snake_head[0]
    y = self.snake_head[1]

    if self.direction == Direction.RIGHT:
        x += 10
    elif self.direction == Direction.LEFT:
        x -= 10
    elif self.direction == Direction.DOWN:
        y += 10
    elif self.direction == Direction.UP:
        y -= 10

    self.snake_head = [x, y]


class SnakeEnv(gym.Env):
    def __init__(self):
        super(SnakeEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(11,), dtype=np.float64
        )
        self.truncated = False

    def render(self):
        cv2.imshow("a", self.img)
        cv2.waitKey(1)
        self.img = np.zeros((500, 500, 3), dtype="uint8")
        # Display Apple
        cv2.rectangle(
            self.img,
            (self.apple_position[0], self.apple_position[1]),
            (self.apple_position[0] + 10, self.apple_position[1] + 10),
            (0, 0, 255),
            3,
        )
        cv2.putText(
            self.img,
            "SCORE: " + str(self.score),
            (350, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        # Display Snake
        for position in self.snake_position:
            cv2.rectangle(
                self.img,
                (position[0], position[1]),
                (position[0] + 10, position[1] + 10),
                (0, 255, 0),
                3,
            )

        # Takes step after fixed time
        t_end = time.time() + 0.01
        k = -1
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(1)
            else:
                continue

    def step(self, action):
        self.prev_actions.append(action)

        # Miramos la posición anterior para saber a donde movernos.
        moveSnake(self, action)

        reward = 0
        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            # print("SE HA COMIDO UNA MANZANA----------------------------")
            self.apple_position, self.score = collision_with_apple(
                self.apple_position, self.score
            )
            self.snake_position.insert(0, list(self.snake_head))
            reward = 10
        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()
            """
            # reward for going in the right direction
            distancia = math.sqrt(
                (self.apple_position[0] - self.snake_head[0]) ** 2
                + (self.apple_position[1] - self.snake_head[1]) ** 2
            )
            reward = (-1 / (distancia + 1)) * 10"""
        # Actualiza la condición de truncado
        self.truncated = collision_with_boundaries(self.snake_head) == 1

        # On collision kill the snake and print the score
        if self.truncated or collision_with_self(self.snake_position) == 1:
            self.done = True

        if self.done:
            reward = -10

        self.reward = reward
        info = {}

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        # Puntos junto a la cabeza.
        point_l = [head_x - 10, head_y]
        point_r = [head_x + 10, head_y]
        point_u = [head_x, head_y - 10]
        point_d = [head_x, head_y + 10]
        # Direcciones.
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
        # Posición manzana.
        apple_x = self.apple_position[0]
        apple_y = self.apple_position[1]

        # create observation:
        observation = [
            (
                dir_r
                and (
                    collision_with_boundaries(point_r)
                    or collision_with_self([point_r] + self.snake_position[1:])
                )
            )
            or (
                dir_l
                and (
                    collision_with_boundaries(point_l)
                    or collision_with_self([point_l] + self.snake_position[1:])
                )
            )
            or (
                dir_u
                and (
                    collision_with_boundaries(point_u)
                    or collision_with_self([point_u] + self.snake_position[1:])
                )
            )
            or (
                dir_d
                and (
                    collision_with_boundaries(point_d)
                    or collision_with_self([point_d] + self.snake_position[1:])
                )
            ),
            (
                dir_u
                and (
                    collision_with_boundaries(point_r)
                    or collision_with_self([point_r] + self.snake_position[1:])
                )
            )
            or (
                dir_d
                and (
                    collision_with_boundaries(point_l)
                    or collision_with_self([point_l] + self.snake_position[1:])
                )
            )
            or (
                dir_r
                and (
                    collision_with_boundaries(point_u)
                    or collision_with_self([point_u] + self.snake_position[1:])
                )
            )
            or (
                dir_l
                and (
                    collision_with_boundaries(point_d)
                    or collision_with_self([point_d] + self.snake_position[1:])
                )
            ),
            (
                dir_d
                and (
                    collision_with_boundaries(point_r)
                    or collision_with_self([point_r] + self.snake_position[1:])
                )
            )
            or (
                dir_u
                and (
                    collision_with_boundaries(point_l)
                    or collision_with_self([point_l] + self.snake_position[1:])
                )
            )
            or (
                dir_l
                and (
                    collision_with_boundaries(point_u)
                    or collision_with_self([point_u] + self.snake_position[1:])
                )
            )
            or (
                dir_r
                and (
                    collision_with_boundaries(point_d)
                    or collision_with_self([point_d] + self.snake_position[1:])
                )
            ),
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            apple_x < head_x,
            apple_x > head_x,
            apple_y < head_y,
            apple_y > head_y,
        ]
        observation = np.array(observation, dtype=int)

        return observation, self.reward, self.done, self.truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.img = np.zeros((500, 500, 3), dtype="uint8")
        # Initial Snake and Apple position
        self.snake_position = [[250, 250], [240, 250], [230, 250]]
        self.apple_position = [
            random.randrange(1, 50) * 10,
            random.randrange(1, 50) * 10,
        ]
        self.score = 0
        self.direction = Direction.RIGHT
        self.snake_head = [250, 250]

        self.done = False

        self.prev_actions = deque(
            maxlen=SNAKE_LEN_GOAL
        )  # however long we aspire the snake to be
        for i in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)  # to create history

        # create observation:
        observation = [0] * 11
        observation = np.array(observation, dtype=int)

        info = {}

        return observation, info
