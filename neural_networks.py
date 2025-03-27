#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 12:27:37 2025

@author: andrey
"""

import torch
import torch.nn as nn


class CardNN(nn.Module):
    def __init__(self, input_size=36 , output_size=37):
        super(CardNN, self).__init__()
        self.fc11 = nn.Linear(input_size , 64) # playing cards
        self.in11 = nn.InstanceNorm1d(64)  # Instance normalization for fc11 layer
        self.fc12 = nn.Linear(1,1) # attacking flag 
        self.fc13 = nn.Linear(input_size , 32) 
        self.in13 = nn.InstanceNorm1d(32)  # Instance normalization for fc13 layer
        self.fc14 = nn.Linear(input_size , 32)
        self.in14 = nn.InstanceNorm1d(32)  # Instance normalization for fc14 layer
        self.fc15 = nn.Linear(1,1) # attacking card index 
        self.fc2 = nn.Linear(130, 128)   # 128 _ dimension of a flag
        self.in2 = nn.InstanceNorm1d(128)  # Instance normalization for fc2 layer
        self.fc3 = nn.Linear(128, output_size)
        self.softmax = nn.Softmax(dim=1)  # Add softmax layer

    def forward(self, x1,x2,x3,x4,x5):
        x1 = torch.relu(self.in11(self.fc11(x1).unsqueeze(0)).squeeze(0))
        x2 = torch.relu(self.fc12(x2))
        x3 = torch.relu(self.in13(self.fc13(x3).unsqueeze(0)).squeeze(0))
        x4 = torch.relu(self.in14(self.fc14(x4).unsqueeze(0)).squeeze(0))
        x5 = self.fc15(x5)
        if len(x1.shape) == 2: x = torch.cat((x1,x2,x3,x4,x5),1)
        if len(x1.shape) == 1: x = torch.cat((x1,x2,x3,x4,x5),0)
        x = torch.relu(self.in2(self.fc2(x).unsqueeze(0)).squeeze(0))
        x = self.fc3(x)
        return self.softmax(x)  # Apply softmax to the output
