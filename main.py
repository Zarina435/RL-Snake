import stable_baselines3 as sb3
from snakeenv import SnakeEnv

env = SnakeEnv()

model = sb3.DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=90000,progress_bar=True)
obs,_=env.reset()

for i in range(10000):
    action,_states=model.predict(obs,deterministic=True)
    obs,rewards,dones,truncate,info=env.step(action)
    env.render()
    if dones:
        break
