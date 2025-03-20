#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 09:04:32 2025

@author: andrey
"""

import matplotlib.pyplot as plt

# Unicode characters for card suits
suit_symbols = {
    'clubs': '♣',
    'diamonds': '♦',
    'hearts': '♥',
    'spades': '♠'
}

def visualize_games(game_log):
    # Print all logs
    print("\n")
    print("\n")
    print("==========================")
    for log_entry in game_log[:3]:
        print("Log Entry:", log_entry)

    # Print all logs
    print("\n")
    print("\n")
    print("==========================")
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    for i, log_entry in enumerate(game_log[:3]):
        ax = axes[i]
        trump_suit = log_entry['trump'][1]  # Extract the suit from the trump tuple
        title = f"Game {log_entry['episode']} Step {log_entry['step']} (Trump: {suit_symbols[trump_suit]})"
        if 'result' in log_entry:
            title += f" - {log_entry['result']}"
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

        # Plot attacking card
        if log_entry['attacker_action']:
            card_text = f"{log_entry['attacker_action'][0]} {suit_symbols[log_entry['attacker_action'][1]]}"
            ax.text(0, 1, card_text, ha='center', va='center', fontsize=12, color='red')

        # Plot defending card
        if log_entry['defender_action']:
            card_text = f"{log_entry['defender_action'][0]} {suit_symbols[log_entry['defender_action'][1]]}"
            ax.text(0, 0, card_text, ha='center', va='center', fontsize=12, color='blue')

        # Plot remaining cards
        remaining_attacker_text = "Remaining Attacker Cards: " + ", ".join([f"{card[0]} {suit_symbols.get(card[1], 'Unknown')}" for card in log_entry['remaining_attacker_hand'] if isinstance(card, tuple) and len(card) == 2])
        remaining_defender_text = "Remaining Defender Cards: " + ", ".join([f"{card[0]} {suit_symbols.get(card[1], 'Unknown')}" for card in log_entry['remaining_defender_hand'] if isinstance(card, tuple) and len(card) == 2])
        ax.text(0, 2, remaining_attacker_text, ha='left', va='center', fontsize=10, color='black')
        ax.text(0, 1.8, remaining_defender_text, ha='left', va='center', fontsize=10, color='black')

    plt.tight_layout()
    plt.show()

# Example game_log for testing
game_log = [
    {
        'episode': 1,
        'step': 1,
        'trump': ('8', 'hearts'),
        'attacker_action': ('10', 'spades'),
        'defender_action': ('6', 'clubs'),
        'remaining_attacker_hand': [('8', 'hearts'), ('6', 'clubs'), ('A', 'diamonds'), ('9', 'hearts'), ('10', 'hearts')],
        'remaining_defender_hand': [('6', 'hearts'), ('8', 'spades'), ('7', 'diamonds'), ('8', 'diamonds'), ('J', 'hearts'), ('6', 'spades'), (('10', 'spades'), 1)],
        'result': 'Attacker wins'
    },
    {
        'episode': 1,
        'step': 1,
        'trump': ('8', 'hearts'),
        'attacker_action': ('10', 'spades'),
        'defender_action': ('6', 'clubs'),
        'remaining_attacker_hand': [('8', 'hearts'), ('6', 'clubs'), ('A', 'diamonds'), ('9', 'hearts'), ('10', 'hearts')],
        'remaining_defender_hand': [('6', 'hearts'), ('8', 'spades'), ('7', 'diamonds'), ('8', 'diamonds'), ('J', 'hearts'), ('6', 'spades'), (('10', 'spades'), 1)]
    },
    {
        'episode': 2,
        'step': 1,
        'trump': ('7', 'clubs'),
        'attacker_action': ('6', 'spades'),
        'defender_action': ('A', 'spades'),
        'remaining_attacker_hand': [('7', 'clubs'), ('K', 'diamonds'), ('9', 'diamonds'), ('8', 'hearts'), ('8', 'diamonds')],
        'remaining_defender_hand': [('Q', 'diamonds'), ('J', 'hearts'), ('K', 'spades'), ('6', 'diamonds'), ('8', 'clubs')],
        'result': 'Defender wins'
    }
]

visualize_games(game_log)