import stable_baselines3 as sb3
from snakeenv import SnakeEnv
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy

env = SnakeEnv()
obs, _ = env.reset()

# Para entrenar.

"""
model = sb3.A2C("MlpPolicy", env, verbose=1)
model = model.learn(total_timesteps=170000, progress_bar=True)
model.save("snake_model_A2C")

model = sb3.A2C.load("snake_model_A2C")"""

# Cargamos los modelos.
model1 = sb3.DQN.load("snake_model_DQN")
# model2 = sb3.A2C.load("snake_model_A2C")
model3 = sb3.PPO.load("snake_model_PPO")

# Evaluamos modelo con DQN.
mean_reward1, std1 = evaluate_policy(model1, env, n_eval_episodes=10)
# Evaluamos modelo con A2C.
# mean_reward2, std2 = evaluate_policy(model2, env, n_eval_episodes=10)
mean_reward2 = 0
std2 = 0
# Evaluamos modelo con PPO.
mean_reward3, std3 = evaluate_policy(model3, env, n_eval_episodes=10)

# Comparamos los resultados.
print(f"Recompensa media DQN: {mean_reward1}")
# print(f"Recompensa media A2C: {mean_reward2}")
print(f"Recompensa media PPO: {mean_reward3}")
print(f"Desviación estándar DQN: {std1}")
# print(f"Desviación estándar A2C: {std2}")
print(f"Desviación estándar PPO: {std3}")

# Mostramos gráfico de recompensa media.
plt.bar(["DQN", "A2C", "PPO"], [mean_reward1, mean_reward2, mean_reward3])
plt.ylabel("Mean Reward")
plt.show()
# Mostramos gráfico de desviación estándar de la recompensa.
plt.bar(["DQN", "A2C", "PPO"], [std1, std2, std3])
plt.ylabel("Mean duration")
plt.show()

# Para visualizar el resultado.
"""
for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, truncate, info = env.step(action)
    env.render()
    if done:
        break"""
