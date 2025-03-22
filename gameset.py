#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 11:32:17 2025

@author: andreyvlasenko
"""
import torch
import torch.nn.functional as F
from neural_networks import attacker_net, defender_net, attacker_optimizer, defender_optimizer
from game_turns import game_turns
import numpy as np



def gameset(game,
            attack_flag,
            defend_flag,
            deck,
            episode, 
            game_log,
            reward_value,
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
    
    # if loss_attacker.grad != None:
    #     loss_attacker.grad.zero()
    #     print("Zeroing Grad")
    # if loss_defender.grad != None:
    #     loss_defender.grad.zero()
        
    counter = 1
    while not big_loop_done:
        print("big_loop_done = ", big_loop_done)
        
        played_cards, reward_attacker, reward_defender, output_defender, output_attacker, game_log, done = game_turns( game, 
                                                                                                      attacker,
                                                                                                      defender,
                                                                                                      attack_flag,
                                                                                                      defend_flag,
                                                                                                      played_cards,
                                                                                                      taken_cards,
                                                                                                      state_attacker,
                                                                                                      state_defender,
                                                                                                      deck,
                                                                                                      episode, 
                                                                                                      game_log,
                                                                                                      attacker_net,
                                                                                                      defender_net,
                                                                                                      reward_value
                                                                                                      )
        big_loop_done = True
        deck_status = game.refill_hands(attacker,defender)
        if deck_status == 0:
            if not game.players[attacker]:
                print("Loop is done")
               # reward_attacker = reward_attacker + 3 * reward_value  # FIXME!
                big_loop_done = True
            if not game.players[defender]:
                print("Loop is done")
               # reward_defender = reward_defender + 3 * reward_value
                big_loop_done = True
            print("big_loop_done = ", big_loop_done)
                
        if any(log_entry['result'] == "Wrong card chosen" for log_entry in game_log): big_loop_done = True
        
        if np.mod(counter,2) == 0:
            attacker = 1
            defender = 0
        else:
            attacker = 0
            defender = 1
        
        
        
        if reward_attacker != 0: reward_attacker = reward_attacker/reward_attacker
        if reward_defender != 0: reward_defender = reward_defender/reward_defender
        
        
        # print("reward_attacker.dtype = ", reward_attacker.dtype)
        # print("reward_defender.dtype = ", reward_defender.dtype)
    
        # Calculate target Q-values using Bellman's equation
        if counter == 1:
            target_attacker = gamma * reward_attacker
            target_defender = gamma * reward_defender            
        else:
            target_attacker = Q_attacker_previous + gamma * reward_attacker
            target_defender = Q_defender_previous + gamma * reward_defender
        
        # print("target_attacker.dtype = ", target_attacker.dtype)
        # print("target_defender.dtype = ", target_defender.dtype)
        # print("Q_attacker_previous.dtype = ", Q_attacker_previous.dtype)
        # print("Q_defender_previous.dtype = ", Q_defender_previous.dtype)
        print(f"Episode: {episode + 1}, Reward Attacker: {reward_attacker}, Reward Defender: {reward_defender}")
        
        # Update states
        Q_attacker_previous = reward_attacker
        Q_attacker_previous = reward_defender
        
        # Calculate loss
        loss_attacker = loss_attacker + F.mse_loss(target_attacker, Q_attacker_previous)
        loss_defender = loss_defender + F.mse_loss(target_defender, Q_defender_previous)
        
    return loss_attacker, loss_defender, game_log