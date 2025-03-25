#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 19:29:49 2025

@author: andreyvlasenko
"""

import torch

def mask_invalid_cards(action_probs, valid_cards, deck):
    # Create a mask for valid cards
    mask = torch.zeros_like(action_probs)
    for card in valid_cards:
        index = deck.index(card)
        mask[0, index] = 1
    # Apply mask to action probabilities
    masked_probs = action_probs * mask
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
        for defend_card in defenders_cards:
            if game.can_beat(attack_card, defend_card) and not game.can_beat(attack_card, chosen_defender_card):
                if verbose: print('Defender can beat, but chosen wrong card')
                defence_decision = "failure"
                break
    else:
        defence_decision = "withdraw"
        
   # print("ATTACK = ", attack_card, "DEFEND = ", chosen_defender_card, "Defenders cards", defenders_cards, "TRUMP ", game.trump_suit  )
    return defence_decision
