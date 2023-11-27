import stable_baselines3 as sb3
from snakeenv import SnakeEnv

env = SnakeEnv()

model = sb3.PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2000000)
