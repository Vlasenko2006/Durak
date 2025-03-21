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
    def __init__(self, input_size=36 , output_size=37):
        super(CardNN, self).__init__()
        self.fc11 = nn.Linear(input_size , 64) # playing cards
        self.fc12 = nn.Linear(1,1) # attacking flag 
        self.fc13 = nn.Linear(input_size , 32) 
        self.fc14 = nn.Linear(input_size , 32)
        self.fc2 = nn.Linear(129, 64)   # 128 _ dimention of a flag
        self.fc3 = nn.Linear(64, output_size)
        self.softmax = nn.Softmax(dim=1)  # Add softmax layer

    def forward(self, x1,x2,x3,x4):
        x1 = torch.relu(self.fc11(x1))
        x2 = torch.relu(self.fc12(x2))
        x3 = torch.relu(self.fc13(x3))
        x4 = torch.relu(self.fc14(x4))
        if len(x1.shape) == 2: x = torch.cat((x1,x2,x3,x4),1)
        if len(x1.shape) == 1: x = torch.cat((x1,x2,x3,x4),0)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)  # Apply softmax to the output

# Initialize neural networks for both players
attacker_net = CardNN()
defender_net = CardNN()

# Optimizers for both networks
attacker_optimizer = optim.Adam(attacker_net.parameters(), lr=0.001)
defender_optimizer = optim.Adam(defender_net.parameters(), lr=0.001)