#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 11:32:17 2025

@author: andreyvlasenko
"""
import torch
import torch.nn.functional as F
from game_turns import game_turns
import numpy as np

#### Must reset cards for ech player

 

def gameset(game,
            players,
            attack_flag,
            defend_flag,
            full_deck,
            episode, 
            game_log,
            reward_value,
            margin_attacker,
            margin_defender,            
            gamma =.99
            ):
    
    game.create_deck()

    
    
    # attacker_ID
    state_attacker = torch.tensor(game.get_state(0), dtype=torch.float32, requires_grad=True).unsqueeze(0)
    state_defender = torch.tensor(game.get_state(1) , dtype=torch.float32, requires_grad=True).unsqueeze(0)
    
    
    attacker = 0
    defender = 1
    
    played_cards = torch.zeros(1,36)
    taken_cards = torch.zeros(2,36)
    Q_attacker_previous = Q_defender_previous = torch.tensor([1.], dtype=torch.float32, requires_grad=True)
    
    ####################  One turn game
    ##################### Need a loop over this turn, :: while deck_status >0 and (game.players[attacker] and game.players[attacker])
    
    big_loop_done = False
    loss_defender = torch.tensor([0.], dtype=torch.float32, requires_grad=True)
    loss_attacker = torch.tensor([0.], dtype=torch.float32, requires_grad=True)
        
    counter = 0
    while not big_loop_done:
        #print("big_loop_done = ", big_loop_done, 'counter = ', counter)
        counter = counter + 1

        
        played_cards, reward_attacker, reward_defender,\
            output_defender, output_attacker,defence_decision , game_log = game_turns( 
                                                                    game,   # output_attacker might be obsolete
                                                                    attacker,
                                                                    defender,
                                                                    attack_flag,
                                                                    defend_flag,
                                                                    played_cards,
                                                                    taken_cards,
                                                                    state_attacker,
                                                                    state_defender,
                                                                    full_deck,
                                                                    episode, 
                                                                    game_log,
                                                                    players[attacker],
                                                                    players[defender],
                                                                    reward_value,
                                                                    counter, 
                                                                    margin_attacker,
                                                                    margin_defender
                                                                    )
        deck_status = game.refill_hands(attacker,defender)
        if deck_status == 0:
            if not game.players[attacker]:
               # print("Loop is done")
                big_loop_done = True
            if not game.players[defender]:
               # print("Loop is done")
                big_loop_done = True
  
        if any(log_entry['result'] == "Wrong card chosen" for log_entry in game_log): big_loop_done = True
        
        if np.mod(counter,2) == 0:
            attacker = 1
            defender = 0
        else:
            attacker = 0
            defender = 1
        
        # Calculate target Q-values using Bellman's equation
        if counter == 1:
            target_attacker = gamma * reward_attacker
            target_defender = gamma * reward_defender            
        else:
            target_attacker = Q_attacker_previous + gamma * reward_attacker
            target_defender = Q_defender_previous + gamma * reward_defender
        
        if defence_decision == "withdraw": counter += 1 # attacker can attack again
        

        # Update states
        Q_attacker_previous = reward_attacker
        Q_attacker_previous = reward_defender
        
        # Calculate loss
        loss_attacker = loss_attacker + F.mse_loss(target_attacker, Q_attacker_previous)
        loss_defender = loss_defender + F.mse_loss(target_defender, Q_defender_previous)
    
    #print("BIG LOOP DONE")
    return loss_attacker, loss_defender, game_log, players