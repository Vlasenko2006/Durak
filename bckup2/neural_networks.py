#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 12:27:37 2025

@author: andrey
"""

import torch
import torch.nn as nn
import torch.optim as optim

class CardNN(nn.Module):
    def __init__(self, input_size=36, output_size=36):
        super(CardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.softmax = nn.Softmax(dim=1)  # Add softmax layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)  # Apply softmax to the output

# Initialize neural networks for both players
attacker_net = CardNN()
defender_net = CardNN()

# Optimizers for both networks
attacker_optimizer = optim.Adam(attacker_net.parameters(), lr=0.001)
defender_optimizer = optim.Adam(defender_net.parameters(), lr=0.001)