#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 06:46:43 2025

@author: andreyvlasenko
"""

def rewards(game_log,
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
            verbose = False):
    
 # attacker gets reward if it forces defender to defend more than one turn       
 reward_attacker = reward_attacker + (step_number - 1)*reward_value  * attacker_card_prob

 
 if defence_decision == "failure":
     if verbose: print("Defence Failure")
     result = "Wrong card chosen"
     done = True
     

 elif defence_decision == "withdraw":
     print("Withdraw")
   #  print("RW = ", reward_value)
     result = "Attacker wins"
     done = True


 if chosen_defender_card not in game.players[defender] or not game.can_beat(chosen_attackers_card, chosen_defender_card):
     done = True
     result = "Attacker wins"
     chosen_defender_card = None
     if verbose: print("Attacker wins from the first turn")
 else:
     game.players[defender].remove(chosen_defender_card)
    # game.players[defender].remove(chosen_defender_card)
     


     # No more cards in hand remaining, defender / attacker wins
     if not game.players[attacker]:
         reward_attacker =  reward_value * attacker_card_prob
         done = True
         result = "Attacker wins, no cards remaining"
         played_cards = not_playing_cards.clone()
     elif not game.players[defender]:
         done = True
         result = "Defender wins, no cards remaining"
         played_cards = not_playing_cards.clone()
     elif not game.players[defender] and not game.players[attacker]:
         done = True
         result = "No winner, no cards remaining"
         played_cards = not_playing_cards.clone()
     else:
         result = "Defender wins 2"
         
 if result != "Wrong card chosen" or result != "Attacker wins":
     reward_defender = reward_defender + reward_value * defender_card_prob 
 

# print(winner)
 
 if verbose: print(f"Episode {episode + 1}:" + result)
 game_log.append({
     'episode': episode + 1,
     'step': counter,
     'trump': game.trump,
     'attacker_action': chosen_attackers_card,
     'defender_action': chosen_defender_card,
     'remaining_attacker_hand': list(game.players[attacker]),
     'remaining_defender_hand': list(game.players[defender]),
     'result': result
 })
 
 #print('attacker_action', chosen_attackers_card,'defender_action', chosen_defender_card,'trump', game.trump)
 return done, played_cards, reward_attacker, reward_defender, game_log
 