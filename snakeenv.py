# Adapted from: https://github.com/TheAILearner/Snake-Game-using-OpenCV-Python/blob/master/snake_game_using_opencv.ipynb
# Get from Sentdex: https://www.youtube.com/watch?v=uKnjGn8fF70&list=PLQVvvaa0QuDf0O2DWwLZBfJeYY-JOeZB1&index=3
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random
import time
from collections import deque

SNAKE_LEN_GOAL = 30


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


class SnakeEnv(gym.Env):
    def __init__(self):
        super(SnakeEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-500, high=500, shape=(5 + SNAKE_LEN_GOAL,), dtype=np.float64
        )
        self.truncated = False
        self.alive_reward = 0.05
        self.prev_dist_to_apple = 0
        self.numero_manzanas = 0

    def step(self, action):
        self.prev_actions.append(action)
        cv2.imshow("a", self.img)
        cv2.waitKey(1)
        self.img = np.zeros((500, 500, 3), dtype="uint8")

        recompensa_colision = 0
        apple_reward = 0
        recompensa_x = 0
        recompensa_y = 0

        # Display Apple
        cv2.rectangle(
            self.img,
            (self.apple_position[0], self.apple_position[1]),
            (self.apple_position[0] + 10, self.apple_position[1] + 10),
            (0, 0, 255),
            3,
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
        t_end = time.time() + 0.05
        k = -1
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(1)
            else:
                continue

        button_direction = action
        # Change the head position based on the button direction.
        # Se comprueba que con la acción no nos chocamos con el cuerpo.
        if button_direction == 1:
            self.snake_head[0] += 10
        elif button_direction == 0:
            self.snake_head[0] -= 10
        elif button_direction == 2:
            self.snake_head[1] += 10
        elif button_direction == 3:
            self.snake_head[1] -= 10

        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(
                self.apple_position, self.score
            )
            self.snake_position.insert(0, list(self.snake_head))
            apple_reward = 200
            self.numero_manzanas += 1
            print("SE HA COMIDO UNA MANZANA!!-------------------------")

        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        # Actualiza la condición de truncado
        self.truncated = collision_with_boundaries(self.snake_head) == 1

        reward_colision = False
        # On collision kill the snake and print the score
        if self.truncated or collision_with_self(self.snake_position) == 1:
            reward_colision = True
            font = cv2.FONT_HERSHEY_SIMPLEX
            self.img = np.zeros((500, 500, 3), dtype="uint8")
            cv2.putText(
                self.img,
                "Your Score is {}".format(self.score),
                (140, 250),
                font,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("a", self.img)

            self.reward = True
            self.done = True

        if reward_colision == True:
            recompensa_colision = -100

        # Distancia euclídea a la manzana.
        euclidean_dist_to_apple = np.linalg.norm(
            np.array(self.snake_head) - np.array(self.apple_position)
        )
        # Distancia de manhattan a la manzana.
        manhattan_dist_to_apple = np.abs(
            self.snake_head[0] - self.apple_position[0]
        ) + np.abs(self.snake_head[1] - self.apple_position[1])

        # Recompensa por la distancia a la manzana.
        recompensa_distancia = 1 / (euclidean_dist_to_apple + 1)
        # recompensa_distancia = 1 / (manhattan_dist_to_apple + 1)
        # Recompensa por acercarse a la manzana.
        reward_acercarse = self.prev_dist_to_apple - recompensa_distancia
        # Recompensa por estar encaminado.
        if self.apple_position[0] == self.snake_head[0]:
            recompensa_x = 2
        if self.apple_position[1] == self.snake_head[1]:
            recompensa_y = 2

        print("Recompensa distancia: ", recompensa_distancia)
        print("Recompensa por acercarse: ", reward_acercarse)
        # Calcular la recompensa total
        # self.total_reward = ((250 - euclidean_dist_to_apple) + apple_reward - 0.1)/100

        # recompensa_total = 0 + recompensa_distancia * 10 + apple_reward
        recompensa_total = (
            (
                -euclidean_dist_to_apple
                + reward_acercarse
                + recompensa_colision
                + recompensa_y
                + recompensa_x
            )
            * (self.numero_manzanas + 1)
            * 10
            / 100
        )

        self.total_reward = recompensa_total

        print("Recompensa total: ", self.total_reward)
        print("---------------------")

        # self.reward = self.total_reward - self.prev_reward
        self.reward = self.total_reward
        self.prev_reward = self.total_reward
        self.prev_dist_to_apple = euclidean_dist_to_apple
        if self.done:
            self.reward = -10
        info = {}

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        # create observation:

        observation = [
            head_x,
            head_y,
            apple_delta_x,
            apple_delta_y,
            snake_length,
        ] + list(self.prev_actions)
        observation = np.array(observation)

        return observation, self.total_reward, self.done, self.truncated, info

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
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [250, 250]

        self.prev_reward = 0
        self.alive_reward = 0.05
        self.prev_dist_to_apple = 0
        self.numero_manzanas = 0

        self.done = False

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        self.prev_actions = deque(
            maxlen=SNAKE_LEN_GOAL
        )  # however long we aspire the snake to be
        for i in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)  # to create history

        # create observation:
        observation = [
            head_x,
            head_y,
            apple_delta_x,
            apple_delta_y,
            snake_length,
        ] + list(self.prev_actions)
        observation = np.array(observation)

        info = {}

        return observation, info
