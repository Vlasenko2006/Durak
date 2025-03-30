import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
from gamer import gamer


# Initialize the gamer instance
gamer0 = gamer()


# Get opponent's attacking card
chosen_attackers_card, attacker_card_index, attack_done = gamer0.opponent_attacks()
#print(f"Opponent's attacking card: {chosen_attackers_card}, Index: {attacker_card_index}, Attack done: {attack_done}")


# if not attack_done:
#     # Get opponent's defending card
#     chosen_defender_card, defend_done = gamer0.opponent_defends(chosen_attackers_card, attacker_card_index)
#     print(f"Opponent's defending card: {chosen_defender_card}, Defense done: {defend_done}")
