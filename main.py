import stable_baselines3 as sb3
from snakeenv import SnakeEnv


env = SnakeEnv()
obs, _ = env.reset()

""""
model = sb3.DQN("MlpPolicy", env, verbose=1)
model = model.learn(total_timesteps=170000, progress_bar=True)
model.save("snake_model2")"""

model = sb3.DQN.load("snake_model2")

for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, truncate, info = env.step(action)
    env.render()
    if done:
        break
