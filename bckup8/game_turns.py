#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 08:50:18 2025

@author: andrey
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



def game_turns(game,
               state_attacker,
               state_defender,
               deck, 
               episode,
               game_log, 
               attacker_net, 
               defender_net
               ):
    step_number = 0
    attack_value = None
    reward_attacker = 0
    reward_defender = 0
    done = False
    attack_cards = []
    defense_cards = []

    while not done:
        step_number += 1

        # Print the current state of the game
        print(f"Episode {episode + 1}, Step {step_number}")
        print(f"Attacker's Hand: {game.players[0]}")
        print(f"Defender's Hand: {game.players[1]}")

        # Attacker's turn
        valid_attacker_cards = [card for card in game.players[0] if card in deck]
        if attack_value:
            valid_attacker_cards = [card for card in valid_attacker_cards if card[0] == attack_value]
        
        while not valid_attacker_cards:
            print(f"Episode {episode + 1}: No valid cards to attack. Drawing a new card.")
            game.draw_card(0)
            valid_attacker_cards = [card for card in game.players[0] if card in deck]
            state_attacker = torch.tensor(game.get_state(0), dtype=torch.float32, requires_grad=True).unsqueeze(0)

        attacker_action_probs = attacker_net(state_attacker)
        masked_attacker_action_probs, mask = mask_invalid_cards(attacker_action_probs, valid_attacker_cards, deck)
        attacker_card_index = torch.argmax(masked_attacker_action_probs).item()
        chosen_card = game.index_to_card(attacker_card_index)

        if chosen_card not in game.players[0]:
            # Invalid move, attacker loses
            reward_attacker = -100
            reward_defender = 100
            done = True
            defender_action_probs = torch.zeros_like(attacker_action_probs, requires_grad=True)  # Initialize defender_action_probs to avoid unbound error
            print(f"Episode {episode + 1}: Invalid move by attacker.")
            game_log.append({
                'episode': episode + 1,
                'step': step_number,
                'trump': game.trump,
                'attacker_action': chosen_card,
                'defender_action': None,
                'remaining_attacker_hand': list(game.players[0]),
                'remaining_defender_hand': list(game.players[1]),
                'result': 'Defender wins'
            })
            return step_number, reward_attacker, reward_defender, attack_cards, defense_cards, game_log, done
        else:
            # Valid move, proceed with defense
            attack_cards.append((chosen_card, step_number))
            game.players[0].remove(chosen_card)
            state_defender = torch.tensor(game.get_state(1), dtype=torch.float32, requires_grad=True).unsqueeze(0)

            if attack_value is None:
                attack_value = chosen_card[0]

            # Defender's turn
            print(f"Episode {episode + 1}: Defender's turn.")
            valid_defender_cards = [card for card in game.players[1] if (card[1] == chosen_card[1] and deck.index(card) > deck.index(chosen_card)) or card[1] == game.trump]
            defender_action_probs = defender_net(state_defender)
            masked_defender_action_probs, _ = mask_invalid_cards(defender_action_probs, valid_defender_cards, deck)
            defender_card_index = torch.argmax(masked_defender_action_probs).item()
            chosen_defender_card = game.index_to_card(defender_card_index)

            if chosen_defender_card not in game.players[1] or not game.can_beat(chosen_card, chosen_defender_card):
                # Invalid move or cannot beat, defender loses
                reward_attacker = 100
                reward_defender = 0
                game.players[1].extend(attack_cards)  # Defender takes all attack cards
                attack_cards = []  # Clear attack cards
                done = True
                print(f"Episode {episode + 1}: Attacker wins. Defender takes all attack cards.")
                game_log.append({
                    'episode': episode + 1,
                    'step': step_number,
                    'trump': game.trump,
                    'attacker_action': chosen_card,
                    'defender_action': chosen_defender_card,
                    'remaining_attacker_hand': list(game.players[0]),
                    'remaining_defender_hand': list(game.players[1]),
                    'result': 'Attacker wins'
                })
            else:
                # Valid defense, proceed with next round
                defense_cards.append(chosen_defender_card)
                print(f"Episode {episode + 1}: Defense Card: {chosen_defender_card}")
                game.players[1].remove(chosen_defender_card)
                state_attacker = torch.tensor(game.get_state(0), dtype=torch.float32, requires_grad=True).unsqueeze(0)

                # Check win conditions and rewards
                if not game.players[0]:
                    reward_attacker = 100
                    reward_defender = -100
                    done = True
                    print(f"Episode {episode + 1}: Attacker wins.")
                    game_log.append({
                        'episode': episode + 1,
                        'step': step_number,
                        'trump': game.trump,
                        'attacker_action': chosen_card,
                        'defender_action': chosen_defender_card,
                        'remaining_attacker_hand': list(game.players[0]),
                        'remaining_defender_hand': list(game.players[1]),
                        'result': 'Attacker wins'
                    })
                elif not game.players[1]:
                    reward_attacker = -100
                    reward_defender = 100
                    done = True
                    print(f"Episode {episode + 1}: Defender wins.")
                    game_log.append({
                        'episode': episode + 1,
                        'step': step_number,
                        'trump': game.trump,
                        'attacker_action': chosen_card,
                        'defender_action': chosen_defender_card,
                        'remaining_attacker_hand': list(game.players[0]),
                        'remaining_defender_hand': list(game.players[1]),
                        'result': 'Defender wins'
                    })

            if not any(card[0] == attack_value for card in game.players[0]):
                attack_value = None
                done = True
                print(f"Episode {episode + 1}: Attacker has no more cards of the same value.")
                game_log.append({
                    'episode': episode + 1,
                    'step': step_number,
                    'trump': game.trump,
                    'attacker_action': chosen_card,
                    'defender_action': chosen_defender_card,
                    'remaining_attacker_hand': list(game.players[0]),
                    'remaining_defender_hand': list(game.players[1]),
                    'result': 'Defender wins'
                })
                break

            # Attacker decides whether to continue attack
            continue_attack = torch.rand(1).item() > 0.15  # Randomly decide to continue or stop
            if not continue_attack:
                done = True
                # Check if defender successfully defended
                if len(defense_cards) == len(attack_cards):
                    reward_attacker = -100
                    reward_defender = 100
                    print(f"Episode {episode + 1}: Defender successfully defended.")
                    game_log.append({
                        'episode': episode + 1,
                        'step': step_number,
                        'trump': game.trump,
                        'attacker_action': chosen_card,
                        'defender_action': chosen_defender_card,
                        'remaining_attacker_hand': list(game.players[0]),
                        'remaining_defender_hand': list(game.players[1]),
                        'result': 'Defender wins'
                    })
                else:
                    reward_attacker = 100
                    reward_defender = 0
                    game.players[1].extend(attack_cards)  # Defender takes all attack cards
                    attack_cards = []  # Clear attack cards
                    print(f"Episode {episode + 1}: Attacker decides to stop the attack. Defender takes all attack cards.")
                    game_log.append({
                        'episode': episode + 1,
                        'step': step_number,
                        'trump': game.trump,
                        'attacker_action': chosen_card,
                        'defender_action': chosen_defender_card,
                        'remaining_attacker_hand': list(game.players[0]),
                        'remaining_defender_hand': list(game.players[1]),
                        'result': 'Attacker wins'
                    })
                break

        # Log the actions and remaining cards after each step
        game_log.append({
            'episode': episode + 1,
            'step': step_number,
            'trump': game.trump,
            'attacker_action': chosen_card,
            'defender_action': chosen_defender_card if 'chosen_defender_card' in locals() else None,
            'remaining_attacker_hand': list(game.players[0]),
            'remaining_defender_hand': list(game.players[1])
        })

        # Print remaining cards after each step
        print(f"Remaining Attacker's Hand: {game.players[0]}")
        print(f"Remaining Defender's Hand: {game.players[1]}")

    return step_number, reward_attacker, reward_defender, attack_cards, defense_cards, game_log, done
