#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 19:29:49 2025

@author: andreyvlasenko
"""

import torch


import torch

#import torch

def mask_invalid_cards(action_probs, valid_cards, deck):
    # Create a mask for valid cards
    mask = torch.zeros_like(action_probs)
    valid_indices = [deck.index(card) for card in valid_cards]
    
    # Convert valid_indices back to cards
#    valid_index_cards = [deck[index] for index in valid_indices]
    
    for index in valid_indices:
        mask[0, index] = 1
    
    masked_probs = action_probs * mask
    # print(f"Debug mask: action_probs = {action_probs}")
    # print(f"Debug mask: valid_cards = {valid_cards}")
    # print(f"Debug mask: valid_indices = {valid_indices}")
    # print(f"Debug mask: valid_index_cards = {valid_index_cards}")
    # print(f"Debug mask: mask = {mask}")
    # print(f"Debug mask: masked_action_probs = {masked_probs}")
    
    # Print all masked cards
    #masked_cards = [deck[i] for i in range(len(deck)) if masked_probs[0, i] > 0]
    # print(f"Debug mask: masked_cards = {masked_cards}")
    # print(f" ==== Mask Debugging finished =====")
    
    # Apply mask to action probabilities

    # Normalize masked probabilities to sum to 1
    if masked_probs.sum().item() != 0:
        masked_probs = masked_probs / masked_probs.sum()
    return masked_probs, mask


def defender_can_beat(defenders_cards, 
                      attack_card,
                      chosen_defender_card,
                      game, 
                      decision_to_defend,
                      margin_defender,
                      verbose=False):
    
  #  print("decision_to_defend , margin_defender", decision_to_defend , margin_defender)
    if decision_to_defend > margin_defender:
        defence_decision = "decide_to_defend"
        can_beat = any(game.can_beat(attack_card, defend_card) for defend_card in defenders_cards)
        if can_beat and not game.can_beat(attack_card, chosen_defender_card):
            if verbose: print('Defender can beat, but chosen wrong card')
            defence_decision = "failure"
        if not can_beat: defence_decision = "wrong_decision"
    else:
        defence_decision = "withdraw"
        
   # print("ATTACK = ", attack_card, "DEFEND = ", chosen_defender_card, "Defenders cards", defenders_cards, "TRUMP ", game.trump_suit  )
    return defence_decision
