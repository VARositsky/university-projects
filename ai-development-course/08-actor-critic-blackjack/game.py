import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Функция предобработки (та же самая, что и при обучении)
def preprocess_state(obs):
    return np.array([obs[0] / 32.0, obs[1] / 11.0, float(obs[2])], dtype=np.float32)

def play_one_game(model_path="blackjack_actor.keras"):
    """Запускает ОДНУ игру с текстовым логом ходов."""
    env = gym.make("Blackjack-v1", sab=True)
    actor = keras.models.load_model(model_path)

    obs, _ = env.reset()
    done = False
    step = 1

    print(f"\n[Старт] Очки: {obs[0]} | Дилер: {obs[1]} | Туз: {'Да' if obs[2] else 'Нет'}")

    while not done:
        state_t = tf.convert_to_tensor([preprocess_state(obs)], dtype=tf.float32)
        probs = actor(state_t, training=False)[0].numpy()
        action = int(np.argmax(probs))
        
        print(f"Ход {step}: {'HIT (Взять)' if action == 0 else 'STICK (Остаться)'} | Уверенность: {probs[action]*100:.1f}%")
        
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1

    print(f"Итог: {obs[0]} очков. Награда: {reward}")
    env.close()

def test_bulk_games(model_path="blackjack_actor.keras", num_games=1000):
    """Тестирует модель на большом количестве игр (без обучения)"""
    env = gym.make("Blackjack-v1", sab=True)
    actor = keras.models.load_model(model_path)
    
    wins = losses = draws = 0
    
    print(f"\nЗапуск {num_games} тестовых игр в фоне...")
    
    for _ in range(num_games):
        obs, _ = env.reset()
        done = False
        
        while not done:
            state_t = tf.convert_to_tensor([preprocess_state(obs)], dtype=tf.float32)
            # Жадный выбор (берем действие с наибольшей вероятностью)
            action = int(np.argmax(actor(state_t, training=False)[0].numpy()))
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
        if reward == 1.0:
            wins += 1
        elif reward == -1.0:
            losses += 1
        else:
            draws += 1
            
    print(f"Побед:     {wins} ({(wins/num_games)*100:.2f}%)")
    print(f"Поражений: {losses} ({(losses/num_games)*100:.2f}%)")
    print(f"Ничьих:    {draws} ({(draws/num_games)*100:.2f}%)")
    
    effective_winrate = (wins / max(1, wins + losses)) * 100
    print(f"Винрейт (без учета ничьих): {effective_winrate:.2f}%\n")
    
    env.close()

if __name__ == "__main__":
    play_one_game()
    test_bulk_games(num_games=1000)