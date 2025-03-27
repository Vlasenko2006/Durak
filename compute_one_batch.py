#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from durak_game import DurakGame
from gameset import gameset



def compute_one_batch(player_0_optimizer,
                      player_1_optimizer,
                      batch_size,
                      players,
                      attack_flag,
                      defend_flag,
                      episode,
                      deck,
                      game_log,
                      reward_value,
                      margin_attacker,
                      margin_defender
                      ):
    # Zero gradients at the start of each batch
    player_0_optimizer.zero_grad()
    player_1_optimizer.zero_grad()
    
    accumulated_loss_defender = torch.tensor([0.], dtype=torch.float32, requires_grad=True)
    accumulated_loss_attacker = torch.tensor([0.], dtype=torch.float32, requires_grad=True)

     
    for batch in range(batch_size):
        game = DurakGame()

    
        loss_attacker_loc, loss_defender_loc, game_log, players = gameset(
                    game,
                    players,
                    attack_flag,
                    defend_flag,
                    deck,
                    episode, 
                    game_log,
                    reward_value,
                    margin_attacker,
                    margin_defender,
                    gamma=0.99
        )
                    
        accumulated_loss_attacker = accumulated_loss_attacker + loss_attacker_loc / batch_size
        accumulated_loss_defender = accumulated_loss_defender + loss_defender_loc / batch_size           
    
   
    
    print(f"num_episodes {episode} Accumulated Losses:", accumulated_loss_attacker, accumulated_loss_defender)

    # Perform backward pass after accumulating the losses
    accumulated_loss_attacker.backward(retain_graph=True)
    accumulated_loss_defender.backward(retain_graph=True)

    # Update networks with accumulated gradients
    player_0_optimizer.step()
    player_1_optimizer.step()
    
    # Remove gradients to free the graph
    # player_0_optimizer = player_0_optimizer.zero_grad()
    # player_1_optimizer = player_1_optimizer.zero_grad()
    # players[0] = players[0].detach()
    # players[1] = players[0].detach()
    
    # Detach the accumulated losses from the computation graph
    accumulated_loss_attacker = accumulated_loss_attacker.detach().numpy()
    accumulated_loss_defender = accumulated_loss_defender.detach().numpy()
    
    
    return game_log, players, accumulated_loss_attacker, accumulated_loss_defender, player_0_optimizer, player_1_optimizer  