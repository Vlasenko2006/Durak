#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 22:12:04 2025

@author: andreyvlasenko
"""

import torch
from check_combinations import mask_invalid_cards, defender_can_beat

def defence(game,
            defender_net,
            state_defender,    # Cards on hand of the defwnder
            defender,          # NN's ID, which defends now. Either 0 or 1. 
            defend_flag,
            played_cards,
            cards_on_a_table,
            chosen_attackers_card,
            attacker_card_index,
            margin_defender,
            full_deck
            ):

    # print("Defence")
    # print('========================================\n')
    attacker_card_index = torch.tensor([[attacker_card_index]], dtype=torch.float32)
    
    
    # convering suits and  values in the defender's hand into indexes by using game.get_state 
    state_defender = torch.tensor(game.get_state(defender), dtype=torch.float32, requires_grad=True).unsqueeze(0)

    output_defender = defender_net(state_defender,
                                   defend_flag,
                                   played_cards,
                                   cards_on_a_table, 
                                   attacker_card_index
                                   )
    
    defender_action_probs = output_defender[..., :-1]
    probability_to_defend = output_defender[..., -1]
    masked_defender_action_probs, _ = mask_invalid_cards(defender_action_probs, game.players[defender], full_deck)
    
    defender_card_index = torch.argmax(masked_defender_action_probs).item()
    chosen_defender_card = game.index_to_card(defender_card_index)
    
    defender_card_prob = masked_defender_action_probs[0, defender_card_index]
    
    # Compute mean of masked_defender_action_probs excluding defender_card_index
    masked_probs_excluding_index = masked_defender_action_probs.clone()
    masked_probs_excluding_index[0, defender_card_index] = 0  # Set the value at defender_card_index to 0
    mean_masked_defender_action_probs = masked_probs_excluding_index[masked_probs_excluding_index != 0].mean()
    
    
    cards_on_a_table = game.update_state(defender, defender_card_index, cards_on_a_table)
    
    game.players[defender].remove(chosen_defender_card) #
    
    defence_decision = defender_can_beat(game.players[defender],
                                         chosen_attackers_card,
                                         chosen_defender_card,
                                         game,
                                         probability_to_defend,
                                         margin_defender
                                         )
    
    # print("Defender cards_on_a_table", cards_on_a_table)
    # print('========================================\n')
    
    
    
    
    if defence_decision == "withdraw":
     #   print("Before removing game.players[defender]", game.players[defender] , "\n")
     #   print("cards_on_a_table", cards_on_a_table, "\n")
        game.players[defender].extend(game.indexes_to_cards(cards_on_a_table))
        cards_on_a_table = []
     #   print("After removing game.players[defender]", game.players[defender], "LEN = ", len(game.players[defender]))
        
      #  print("==========================\n")
    
    return cards_on_a_table, defender_card_prob, defence_decision, chosen_defender_card, output_defender, probability_to_defend, mean_masked_defender_action_probs