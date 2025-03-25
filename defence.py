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
            state_defender,
            defender,
            defend_flag,
            played_cards,
            cards_on_a_table,
            chosen_attackers_card,
            attacker_card_index,
            margin_defender,
            deck
            ):

    
         attacker_card_index = torch.tensor([[attacker_card_index]], dtype=torch.float32)
    
         state_defender = torch.tensor(game.get_state(1), dtype=torch.float32, requires_grad=True).unsqueeze(0)

         output_defender = defender_net(state_defender,
                                        defend_flag,
                                        played_cards,
                                        cards_on_a_table, 
                                        attacker_card_index
                                        )
         
         defender_action_probs = output_defender[..., :-1]
         decision_to_defend = output_defender[..., -1]
         masked_defender_action_probs, _ = mask_invalid_cards(defender_action_probs, game.players[defender], deck)
         defender_card_index = torch.argmax(masked_defender_action_probs).item()
         chosen_defender_card = game.index_to_card(defender_card_index)
         
         
         defender_card_prob = masked_defender_action_probs[0,defender_card_index]
         
         
         
    
    
         cards_on_a_table = game.update_state(defender, defender_card_index, cards_on_a_table)
    
         defence_decision = defender_can_beat(game.players[defender],
                                              chosen_attackers_card,
                                              chosen_defender_card,
                                              game,
                                              decision_to_defend,
                                              margin_defender
                                              )
         
         return cards_on_a_table, defender_card_prob, defence_decision, chosen_defender_card, output_defender