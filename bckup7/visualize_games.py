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

    print("\n")
    print("\n")
    print("==========================")
    
    # Group game logs by episode
    episodes = {}
    for log_entry in game_log[:3]:
        episode = log_entry['episode']
        if episode not in episodes:
            episodes[episode] = []
        episodes[episode].append(log_entry)

    fig, axes = plt.subplots(len(episodes), 1, figsize=(15, 5 * len(episodes)))

    if len(episodes) == 1:
        axes = [axes]  # Ensure axes is iterable

    for ax, (episode, logs) in zip(axes, episodes.items()):
        ax.set_title(f"Game {episode}", y=1.05)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1, len(logs) + 1)
        ax.set_ylim(-1.5, 1)

        for i, log_entry in enumerate(logs):
            trump_suit = log_entry['trump'][1]  # Extract the suit from the trump tuple
            title = f"Step {log_entry['step']} (Trump: {suit_symbols[trump_suit]})"
            if 'result' in log_entry and i == len(logs) - 1:
                result_color = 'red' if 'Attacker wins' in log_entry['result'] else 'blue'
                title += f" - {log_entry['result']}"
                ax.text(i, 0.8, title, ha='center', va='center', fontsize=10, color=result_color)
            else:
                ax.text(i, 0.8, title, ha='center', va='center', fontsize=10, color='black')

            # Plot attacking card
            if log_entry['attacker_action']:
                card_text = f"{log_entry['attacker_action'][0]} {suit_symbols[log_entry['attacker_action'][1]]}"
                ax.text(i, 0.3, card_text, ha='center', va='center', fontsize=10, color='red')

            # Plot defending card
            if log_entry['defender_action']:
                card_text = f"{log_entry['defender_action'][0]} {suit_symbols[log_entry['defender_action'][1]]}"
                ax.text(i, -0.3, card_text, ha='center', va='center', fontsize=10, color='blue')

        # Plot remaining cards below the game
        last_log_entry = logs[-1]
        remaining_attacker_text = "Remaining Attacker Cards: " + ", ".join([f"{card[0]} {suit_symbols.get(card[1], 'Unknown')}" for card in last_log_entry['remaining_attacker_hand'] if isinstance(card, tuple) and len(card) == 2])
        remaining_defender_text = "Remaining Defender Cards: " + ", ".join([f"{card[0]} {suit_symbols.get(card[1], 'Unknown')}" for card in last_log_entry['remaining_defender_hand'] if isinstance(card, tuple) and len(card) == 2])
        ax.text(0, -1, remaining_attacker_text, ha='left', va='center', fontsize=8, color='black')
        ax.text(0, -1.2, remaining_defender_text, ha='left', va='center', fontsize=8, color='black')

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
        'remaining_defender_hand': [('6', 'hearts'), ('8', 'spades'), ('7', 'diamonds'), ('8', 'diamonds'), ('J', 'hearts'), ('6', 'spades')],
        'result': 'Attacker wins'
    },
    {
        'episode': 1,
        'step': 2,
        'trump': ('8', 'hearts'),
        'attacker_action': ('9', 'hearts'),
        'defender_action': ('J', 'hearts'),
        'remaining_attacker_hand': [('8', 'hearts'), ('6', 'clubs'), ('A', 'diamonds'), ('10', 'hearts')],
        'remaining_defender_hand': [('6', 'hearts'), ('8', 'spades'), ('7', 'diamonds'), ('8', 'diamonds'), ('6', 'spades')],
        'result': 'Defender wins'
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

#visualize_games(game_log)