#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 09:32:17 2025

@author: andrey
"""

import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Define the neural network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Hyperparameters
state_size = 4
action_size = 2
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
memory_size = 10000
target_update_freq = 10

# Initialize environment, neural networks, and replay buffer
env = gym.make('CartPole-v1')
policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = deque(maxlen=memory_size)

# Function to select an action
def select_action(state):
    global epsilon
    if np.random.rand() < epsilon:
        return random.randrange(action_size)
    else:
        with torch.no_grad():
            return torch.argmax(policy_net(state)).item()

# Function to update the policy network
def update_policy():
    if len(memory) < batch_size:
        return

    minibatch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    current_q_values = policy_net(states).gather(1, actions).squeeze(1)
    next_q_values = target_net(next_states).max(1)[0]
    target_q_values = rewards + (gamma * next_q_values * (1 - dones))

    loss = nn.MSELoss()(current_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()[0]  # Extract the state from the tuple
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    total_reward = 0
    done = False

    while not done:
        action = select_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        update_policy()

    epsilon = max(epsilon_min, epsilon_decay * epsilon)

    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

print("Training completed!")