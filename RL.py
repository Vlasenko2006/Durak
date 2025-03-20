import gym
import numpy as np


env = gym.make('CartPole-v1')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
q_table = np.zeros((state_size, action_size))

epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.8
discount_rate = 0.95

def update_q_table(state, action, reward, next_state, done):
    best_next_action = np.argmax(q_table[next_state])
    td_target = reward + discount_rate * q_table[next_state][best_next_action] * (1 - done)
    q_table[state][action] += learning_rate * (td_target - q_table[state][action])

num_episodes = 1000
max_steps_per_episode = 100

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    
    for step in range(max_steps_per_episode):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit
        
        next_state, reward, done, _ = env.step(action)
        
        update_q_table(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    epsilon = max(epsilon_min, epsilon_decay * epsilon)
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

print("Training completed!")



import matplotlib.pyplot as plt

rewards = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    
    for step in range(max_steps_per_episode):
        action = np.argmax(q_table[state])
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    rewards.append(total_reward)

plt.plot(range(num_episodes), rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.show()