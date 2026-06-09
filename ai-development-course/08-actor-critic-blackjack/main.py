from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

env = gym.make("Blackjack-v1", sab=True)

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


OBSERVATION_DIM = 3
ACTION_COUNT = 2

HIDDEN_LAYER_SIZES = [128, 64]

LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99
EPISODES_COUNT = 500
PRINT_EVERY = 50


class BlackjackAgent:
    def __init__(self, env):
        self.env = env
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    def _build_actor(self) -> keras.Model:
        model = keras.Sequential(name="policy_network")
        model.add(layers.InputLayer(input_shape=(OBSERVATION_DIM,)))
        for units in HIDDEN_LAYER_SIZES:
            model.add(layers.Dense(units, activation="relu"))
        model.add(layers.Dense(ACTION_COUNT, activation="softmax"))
        return model

    def _build_critic(self) -> keras.Model:
        model = keras.Sequential(name="value_network")
        model.add(layers.InputLayer(input_shape=(OBSERVATION_DIM,)))
        for units in HIDDEN_LAYER_SIZES:
            model.add(layers.Dense(units, activation="relu"))
        model.add(layers.Dense(1))
        return model

    def get_action(self, obs, epsilon=0.0):
        if np.random.random() < epsilon:
            return np.random.choice(ACTION_COUNT)
        else:
            obs_tensor = tf.convert_to_tensor([obs], dtype=tf.float32)
            probs = self.actor(obs_tensor, training=False)[0].numpy()
            return int(np.argmax(probs))

    def train_on_episode(self, states, actions, rewards):
        returns = self.compute_returns(rewards)

        states_t = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_t = tf.convert_to_tensor(actions, dtype=tf.int32)
        returns_t = tf.convert_to_tensor(returns, dtype=tf.float32)

        with tf.GradientTape() as tape:
            action_probs = self.actor(states_t, training=True)
            state_values = self.critic(states_t, training=True)
            state_values = tf.squeeze(state_values, axis=-1)

            critic_loss = tf.reduce_mean(tf.square(returns_t - state_values))
            advantage = returns_t - tf.stop_gradient(state_values)

            indices = tf.stack([tf.range(tf.shape(actions_t)[0]), actions_t], axis=1)
            chosen_log_probs = tf.math.log(tf.gather_nd(action_probs, indices) + 1e-8)
            actor_loss = -tf.reduce_mean(chosen_log_probs * advantage)

            total_loss = actor_loss + critic_loss

        grads = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))

        mean_value = tf.reduce_mean(state_values).numpy()
        return total_loss.numpy(), mean_value

    def compute_returns(self, rewards):
        """Дисконтированная сумма наград"""
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + DISCOUNT_FACTOR * G
            returns.insert(0, G)
        returns = np.array(returns, dtype=np.float32)
        if len(returns) > 1 and np.std(returns) > 1e-6:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        return returns

def preprocess_state(obs):
    player_sum, dealer_card, usable_ace = obs
    return np.array([
        player_sum / 32.0,      
        dealer_card / 11.0,     
        float(usable_ace)       
    ], dtype=np.float32)

def train_agent(env, agent, episodes=EPISODES_COUNT):
    episode_rewards = []
    value_history = []
    loss_history = []

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = (epsilon - epsilon_min) / (episodes * 0.5)

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        state = preprocess_state(obs)
        
        states = []
        actions = []
        rewards = []
        raw_obs_history = [obs]  # Сохраняем оригинальные кортежи состояний для логов
        total_reward = 0

        while True:
            action = agent.get_action(state, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = preprocess_state(next_obs)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            raw_obs_history.append(next_obs)
            total_reward += reward

            state = next_state
            if done:
                break

        loss, mean_val = agent.train_on_episode(states, actions, rewards)
        loss_history.append(loss)
        value_history.append(mean_val)
        episode_rewards.append(total_reward)

        epsilon = max(epsilon_min, epsilon - epsilon_decay)

        if ep % PRINT_EVERY == 0:
            avg_reward = np.mean(episode_rewards[-PRINT_EVERY:])
            print(f"\nЭпизод {ep:5d} | Награда (ср. за {PRINT_EVERY}): {avg_reward:.2f} | Loss: {loss:.4f} | Value: {mean_val:.2f} | Epsilon: {epsilon:.2f}")
            print(f"Лог игры:")
            for i in range(len(actions)):
                action_name = "HIT (Взять)" if actions[i] == 0 else "STICK (Остаться)"
                print(f"Ход {i+1}: Карты {raw_obs_history[i]} -> Действие: {action_name} -> Награда: {rewards[i]}")
            print(f"Финальное состояние: {raw_obs_history[-1]} | Итоговая награда: {total_reward}\т")

    return episode_rewards, value_history, loss_history

def evaluate_agent(env, agent, num_games=100):
    """Запускает num_games эпизодов без обучения"""
    wins = 0
    losses = 0
    draws = 0
    for _ in range(num_games):
        obs, _ = env.reset()
        done = False
        while not done:
            state = preprocess_state(obs)
            obs_t = tf.convert_to_tensor([state], dtype=tf.float32)
            probs = agent.actor(obs_t, training=False)[0].numpy()
            action = int(np.argmax(probs))

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        if reward == 1:
            wins += 1
        elif reward == -1:
            losses += 1
        else:
            draws += 1
    return wins, losses, draws

def plot_value_history(value_history):
    """График изменения value-функции во время обучения"""
    plt.figure(figsize=(10, 5))
    plt.plot(value_history, alpha=0.7, label='Среднее V(s) по эпизоду')
    window = 20
    if len(value_history) > window:
        smooth = np.convolve(value_history, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(value_history)), smooth, 'r-', linewidth=2, label=f'Сглажено (окно {window})')
    plt.xlabel('Эпизод')
    plt.ylabel('Среднее значение V(s)')
    plt.title('Динамика value-функции в процессе обучения')
    plt.legend()
    plt.grid(True)
    plt.show()

env = gym.make("Blackjack-v1", sab=True)

agent = BlackjackAgent(env)

print("Начало обучения Actor-Critic для Blackjack...")
rewards, values, losses = train_agent(env, agent, episodes=EPISODES_COUNT)
print("Обучение завершено.\n")

wins, losses, draws = evaluate_agent(env, agent, num_games=100)
print("Результаты на 100 играх (без обучения)")
print(f"Побед: {wins}, Поражений: {losses}, Ничьих: {draws}")
print(f"Процент побед: {wins/100*100:.1f}%")

# График value-функции
plot_value_history(values)

agent.actor.save("blackjack_actor.keras")
print("Модель актора успешно сохранена в файл 'blackjack_actor.keras'!")