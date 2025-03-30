#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 18:59:34 2025

@author: andreyvlasenko
"""

import torch
from check_combinations import mask_invalid_cards


def attack(attacker_net, 
           attacker,
           attack_value, 
           game,
           attack_flag,
           played_cards,
           cards_on_a_table,
           deck, 
           episode, 
           verbose = False
           ):
    
    
    
    # Attacker's turn
    done = False
    print("Opponent attacks cards before filtering= ", game.players[attacker])
    # Get attackers cards (suit,value)     
    valid_attacker_cards = [card for card in game.players[attacker] if card in deck]
    #print("Opponent attacks cards = ", valid_attacker_cards)

    
    # attack_value is a card chosen for attack (provided that one attack was already done)
    # here weensure that the attacker can only play cards that match the ongoing attack value, 
    # maintaining the rules of the game. To prevent the case that NN chooses the card it does not posses
    if attack_value:
        valid_attacker_cards = [card for card in valid_attacker_cards if card[0] == attack_value]
   
    if not valid_attacker_cards:
        done = True
        if verbose: print(f"Episode {episode + 1}: No valid cards to attack.")

    print("Opponent attacks cards = ", valid_attacker_cards)

    
    empty_index = torch.tensor([[-1]], dtype=torch.float32)
    state_attacker = torch.tensor(game.get_state(0), dtype=torch.float32, requires_grad=True).unsqueeze(0)

    output_attacker = attacker_net(state_attacker,
                                   attack_flag, 
                                   played_cards, 
                                   cards_on_a_table,
                                   empty_index)
    
    attacker_action_probs = output_attacker[..., :-1]
    decision_to_continue_attack = output_attacker[..., -1]

    masked_attacker_action_probs, mask = mask_invalid_cards(attacker_action_probs,
                                                            valid_attacker_cards,
                                                            deck)
    
    attacker_card_index = torch.argmax(masked_attacker_action_probs).item()
    chosen_attackers_card = game.index_to_card(attacker_card_index)
    
    # for i,j in enumerate(masked_attacker_action_probs):
    #     print(game.index_to_card(i))
    # print("Attackers card", chosen_attackers_card )
    
    # for i,j in enumerate(state_attacker):
    #     print(game.index_to_card(i))
        
    # print("attacker_card_index = ", attacker_card_index, "masked_attacker_action_probs", masked_attacker_action_probs)
    attacker_card_prob = masked_attacker_action_probs[0,attacker_card_index]
    
    cards_on_a_table = game.update_state(attacker,
                                         attacker_card_index,
                                         cards_on_a_table)

    if chosen_attackers_card not in game.players[attacker]:
        # Invalid move, attacker loses
        done = True

        if attack_value is None:
            attack_value = chosen_attackers_card[0]
            
    game.players[attacker].remove(chosen_attackers_card)       
    return decision_to_continue_attack,attacker_card_prob, chosen_attackers_card, \
        attacker_card_index, cards_on_a_table, done, output_attacker

