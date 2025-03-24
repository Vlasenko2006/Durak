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

#### Must reset cards for ech player






deck = [('8', 'clubs'),
 ('7', 'spades'),
 ('J', 'clubs'),
 ('A', 'spades'),
 ('J', 'hearts'),
 ('Q', 'clubs'),
 ('7', 'clubs'),
 ('9', 'diamonds'),
 ('9', 'clubs'),
 ('10', 'clubs'),
 ('J', 'spades'),
 ('K', 'clubs'),
 ('10', 'hearts'),
 ('K', 'diamonds'),
 ('8', 'diamonds'),
 ('Q', 'diamonds'),
 ('6', 'clubs'),
 ('Q', 'spades'),
 ('9', 'hearts'),
 ('6', 'spades'),
 ('7', 'diamonds'),
 ('Q', 'hearts'),
 ('7', 'hearts'),
 ('8', 'spades'),
 ('A', 'clubs'),
 ('8', 'hearts'),
 ('6', 'hearts'),
 ('A', 'hearts'),
 ('10', 'diamonds'),
 ('K', 'hearts'),
 ('9', 'spades'),
 ('6', 'diamonds'),
 ('10', 'spades'),
 ('J', 'diamonds'),
 ('A', 'diamonds'),
 ('K', 'spades')]


 

def gameset(game,
            attack_flag,
            defend_flag,
            deck,
            episode, 
            game_log,
            reward_value,
            margin_attacker,
            margin_defender,            
            gamma =.99
            ):
    
    #game.create_deck()
    #game.deal_cards()
    
    
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
        
    counter = 0
    while not big_loop_done:
        #print("big_loop_done = ", big_loop_done, 'counter = ', counter)
        counter = counter + 1
        
        played_cards, reward_attacker, reward_defender, output_defender, output_attacker, game_log = game_turns( game, 
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
                                                                                                      reward_value,
                                                                                                      margin_attacker,
                                                                                                      margin_defender
                                                                                                      )
        deck_status = game.refill_hands(attacker,defender)
        if deck_status == 0:
            if not game.players[attacker]:
                # reward_attacker = reward_attacker + 3 * reward_value  # FIXME!
                print("Loop is done")
                big_loop_done = True
            if not game.players[defender]:
                print("Loop is done")
                # reward_defender = reward_defender + 3 * reward_value
                big_loop_done = True
                
        if any(log_entry['result'] == "Wrong card chosen" for log_entry in game_log): big_loop_done = True
        
        if np.mod(counter,2) == 0:
            attacker = 1
            defender = 0
        else:
            attacker = 0
            defender = 1
        
        #print("deck = ", deck)
        #print("big_loop_done = ", big_loop_done, 'counter = ', counter) 
        if reward_attacker != 0: reward_attacker = reward_attacker/reward_attacker
        if reward_defender != 0: reward_defender = reward_defender/reward_defender
        #big_loop_done = True
        
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
        # print(f"Episode: {episode + 1}, Reward Attacker: {reward_attacker}, Reward Defender: {reward_defender}")
        
        # Update states
        Q_attacker_previous = reward_attacker
        Q_attacker_previous = reward_defender
        
        # Calculate loss
        loss_attacker = loss_attacker + F.mse_loss(target_attacker, Q_attacker_previous)
        loss_defender = loss_defender + F.mse_loss(target_defender, Q_defender_previous)
        
    return loss_attacker, loss_defender, game_log