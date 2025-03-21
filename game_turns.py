#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 08:50:18 2025

@author: andrey
"""
#  Key trump card is in attacker card, we must put it in the last place
#  Атакующий и защитник должны видеть ситуацию на столе и принимать решения об атаке - защите.
#  Нужно расширить ввод нейронной сети на то, что она видит на столе.
#  Возможно добавить LSTM для анализа отбоя или сохранять вектор состояния после кажого хода
#  и передавать его с глвым ходом



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
                      can_beat,
                      decision_to_defend
                      ):
    
    if decision_to_defend > 0.5:
        reward = 100
        for defend_card in defenders_cards:
            if can_beat(attack_card, defend_card) and not can_beat(attack_card, chosen_defender_card): 
                print('Defender can beat, but chosen wrong card')
                reward = -100
                break
    else: 
        reward = 0
    return reward



def game_turns( game, 
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
              game_data = []
              ):

    attack_value = None
    reward_attacker = 0
    reward_defender = 0
    done = False

    # Store the cards for visualization
    attack_cards = []
    defense_cards = []

    cards_on_a_table = torch.zeros(1,36)
    not_playing_cards = played_cards
    
    ###########
    
    
    defender_action_probs = None  # Initialize defender_action_probs
    step_number = 0  # Initialize step number


    while not done:
        step_number += 1  # Increment step number

        # Attacker's turn
        valid_attacker_cards = [card for card in game.players[attacker] if card in deck]
        if attack_value:
            valid_attacker_cards = [card for card in valid_attacker_cards if card[0] == attack_value]
        
        if not valid_attacker_cards:
            done = True
            print(f"Episode {episode + 1}: No valid cards to attack.")
            break
        
        not_playing_cards = not_playing_cards + cards_on_a_table
        for card in not_playing_cards[0,:]:
            assert card == 1. or card == 0., print("Wrong nonpalying cards!")

        print(state_attacker.shape)
        print(not_playing_cards.shape)
        print(cards_on_a_table.shape)
        
        output_attacker = attacker_net(state_attacker, attack_flag, not_playing_cards,cards_on_a_table)
        attacker_action_probs = output_attacker[...,:-1]
        print("attacker_action_probs.shape = ",attacker_action_probs.shape)
        print("output_attacker.shape = ",output_attacker.shape)
        decision_to_continue_attack = output_attacker[...,-1]
        
        masked_attacker_action_probs, mask = mask_invalid_cards(attacker_action_probs, valid_attacker_cards, deck)
        attacker_card_index = torch.argmax(masked_attacker_action_probs).item()
        chosen_card = game.index_to_card(attacker_card_index)
        

        if chosen_card not in game.players[attacker]:
            # Invalid move, attacker loses
            reward_attacker = -100
            reward_defender = 100
            done = True
            defender_action_probs = torch.zeros_like(attacker_action_probs, requires_grad=True)  # Initialize defender_action_probs to avoid unbound error
            print(f"Episode {episode + 1}: Invalid move by attacker.")
            return  # Stop the training loop if an invalid card is chosen
        else:
            # Valid move, proceed with defense
            attack_cards.append((chosen_card, step_number))
            game.players[attacker].remove(chosen_card)
            state_defender = torch.tensor(game.get_state(1), dtype=torch.float32, requires_grad=True).unsqueeze(0)
            print("Chosen_attacker's_card = ",chosen_card)
            cards_on_a_table = game.updage_state(attacker_card_index, cards_on_a_table)

            if attack_value is None:
                attack_value = chosen_card[0]

            # Defender's turn

            print(f"Episode {episode + 1}: Defender's turn.")
            output_defender= defender_net(state_defender,defend_flag,not_playing_cards, cards_on_a_table)
            defender_action_probs = output_defender[...,:-1]
            decision_to_defend = output_defender[...,-1]
            masked_defender_action_probs, _ = mask_invalid_cards(defender_action_probs, game.players[defender], deck)
            defender_card_index = torch.argmax(masked_defender_action_probs).item()
            chosen_defender_card = game.index_to_card(defender_card_index)
            
            cards_on_a_table = game.updage_state(defender_card_index, cards_on_a_table)
            
            decision_to_defend = 1  # For debugging only
            reward = defender_can_beat(game.players[defender], chosen_card, chosen_defender_card, game.can_beat, decision_to_defend)
            if reward == -100:
                reward_defender = reward
                game_log.append({
                    'episode': episode + 1,
                    'step': step_number,
                    'trump': game.trump,
                    'attacker_action': chosen_card,
                    'defender_action': chosen_defender_card,
                    'remaining_attacker_hand': list(game.players[attacker]),
                    'remaining_defender_hand': list(game.players[defender]),
                    'result': "Wrong  card chosen "
                })
                done = True
                break
            
            print( "Attacker's cards",list(game.players[attacker]))
            print( "Defender's cards",list(game.players[defender]))     
            print( "Can beat",game.can_beat(chosen_card, chosen_defender_card)) 
            print("Checking Conditions:")
            if chosen_defender_card not in game.players[defender] or not game.can_beat(chosen_card, chosen_defender_card): 
                print("chosen_defender_card not in game.players[defender]")
            else:
                print("CARD IN THE LIST AND CAN BEAT")
                
            print("chosen_defender_card", chosen_defender_card)
            print("game.players[defender]", game.players[defender])
                
            

            if chosen_defender_card not in game.players[defender] or not game.can_beat(chosen_card, chosen_defender_card):
                # Invalid move or cannot beat, defender loses
                reward_attacker = 100
                reward_defender = -100
                done = True
                winner =  "Attacker wins"
                chosen_defender_card = None
                print("Attacker wins from the first turn")
            else:
                # Valid defense, proceed with next round
                defense_cards.append(chosen_defender_card)
                print(f"Episode {episode + 1}: Defense Card: {chosen_defender_card}")
                game.players[defender].remove(chosen_defender_card)
                state_attacker = torch.tensor(game.get_state(0), dtype=torch.float32, requires_grad=True).unsqueeze(0)

                # No more cards in hand remaining, defender / attacker wins
                if not game.players[attacker]:
                    reward_attacker = 100
                    reward_defender = -100
                    done = True
                    winner = "Attacker wins"
                    played_cards =  not_playing_cards
                elif not game.players[defender]:
                    reward_attacker = -100
                    reward_defender = 100
                    done = True
                    winner = "Defender wins"
                    played_cards =  not_playing_cards
                elif not game.players[defender] and not game.players[attacker]:
                    reward_attacker = 100
                    reward_defender = 100
                    done = True
                    winner = "No winner"
                    played_cards=  not_playing_cards
                else:
                    winner =  "Defender wins"
                    reward_attacker = 100
                    reward_defender = -100
                    
                    
                 #   print("odd situation", chosen_card, chosen_defender_card, game.trump)
            print(f"Episode {episode + 1}:" + winner)
            game_log.append({
                'episode': episode + 1,
                'step': step_number,
                'trump': game.trump,
                'attacker_action': chosen_card,
                'defender_action': chosen_defender_card,
                'remaining_attacker_hand': list(game.players[attacker]),
                'remaining_defender_hand': list(game.players[defender]),
                'result': winner
            })

            if not any(card[0] == attack_value for card in game.players[attacker]):
                attack_value = None
                done = True
                print(f"Episode {episode + 1}: Attacker has no more cards of the same value.")
                break

            # Attacker decides whether to continue attack
            continue_attack = decision_to_continue_attack > 0.0   # Randomly decide to continue or stop
            if not continue_attack:
                played_cards =  not_playing_cards
                done = True
                print(f"Episode {episode + 1}: Attacker decides to stop the attack.")
                break




    return played_cards, reward_attacker, reward_defender, attack_cards, defense_cards, game_log, done