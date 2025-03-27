#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 15:17:28 2025

@author: andreyvlasenko
"""

import torch
from attack import attack
from defence import defence
from rewards import rewards



def game_turn(game, 
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
               counter,
               margin_attacker = 0.5,
               margin_defender = 0.5,
               verbose=False
               ):

    attack_value = None
    reward_attacker = torch.tensor([0.], dtype=torch.float32, requires_grad=True)
    reward_defender = torch.tensor([0.], dtype=torch.float32, requires_grad=True)
    done = False

    cards_on_a_table = torch.zeros(1, 36, dtype=torch.float32)  # Convert to Float
    not_playing_cards = played_cards.clone().float()  # Convert to Float to avoid type mismatch

    step_number = 0  # Initialize step number

    while not done:
         step_number += 1  # Increment step number
         
         
         decision_to_continue_attack,attacker_card_prob,\
             chosen_attackers_card, attacker_card_index, \
                 cards_on_a_table, cards_on_a_table, done, \
                     output_attacker = attack(attacker_net, 
                            attacker,
                            attack_value, 
                            game,
                            attack_flag,
                            played_cards,
                            cards_on_a_table,
                            deck, 
                            episode, 
                            verbose = False
                            )

         # Valid move, proceed with defense
         
         cards_on_a_table, defender_card_prob, \
         defence_decision, chosen_defender_card, \
         output_defender = defence(game,
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
                     )


         if not done:
             done, played_cards, reward_attacker, \
                 reward_defender, game_log=\
                 rewards(game_log,
                         game,
                         reward_value,
                         defence_decision,
                         defender_card_prob,
                         attacker_card_prob,
                         episode,
                         step_number,
                         chosen_attackers_card,
                         chosen_defender_card,
                         attacker,
                         defender,
                         not_playing_cards,
                         reward_attacker,
                         reward_defender,
                         played_cards,
                         done,
                         counter,
                         verbose = False)


    
         if not any(card[0] == attack_value for card in game.players[attacker]):
             attack_value = None
             done = True
             if verbose: print(f"Episode {episode + 1}: Attacker has no more cards of the same value.")
             break
    
         # Attacker decides whether to continue attack
         continue_attack = decision_to_continue_attack > margin_attacker  # Randomly decide to continue or stop
         if not continue_attack:
             played_cards = not_playing_cards.clone()
             done = True
             if verbose: print(f"Episode {episode + 1}: Attacker decides to stop the attack.")
             break
         # if winner == "Defender wins": reward_defender 
         # if winner == "Attacker wins": reward_attacker


        # Update not_playing_cards to include cards on the table without duplication
         not_playing_cards = torch.logical_or(not_playing_cards, cards_on_a_table)
   # print("RFEWARD DEFENDER = ", reward_defender, "RFEWARD ATTACKER= ", reward_attacker)
    return played_cards, reward_attacker, reward_defender, output_defender, output_attacker, game_log