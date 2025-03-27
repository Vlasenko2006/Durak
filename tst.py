#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 13:16:17 2025

@author: andrey
"""

import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
from neural_networks import attacker_net
import torch
from durak_game import DurakGame
# Define the cards for the upper and lower rows
card_top = [
    ("6", "clubs"),
    ("K", "diamonds"),
    ("Q", "hearts"),
    ("J", "spades"),
    ("7", "hearts"),
    ("10", "spades"),
    ("K", "hearts")
]

cards = [
    ("5", "clubs"),
    ("J", "diamonds"),
    ("A", "hearts"),
    ("Q", "spades"),
    ("K", "hearts"),
    ("K", "spades"),
    ("Q", "hearts")
]

# Unicode characters for card suits
suit_symbols = {
    'clubs': '♣',
    'diamonds': '♦',
    'hearts': '♥',
    'spades': '♠'
}

# Colors for suits
suit_colors = {
    'clubs': 'black',
    'diamonds': 'red',
    'hearts': 'red',
    'spades': 'black'
}


game = DurakGame()
deck = game.create_deck()
player0 = attacker_net
player0.load_state_dict(torch.load("attacker1_1100"))


opponents_cards = game.players[0]
my_cards = game.players[1]

deck_status = game.refill_hands(opponents_cards,my_cards)
# if deck_status == 0:
#     if not game.players[0]:
#         winner = "You Lost"
#     if not game.players[1]:
#         winner = "You Win"



        
        # Gameplay settings
# self.deck_is_empty = deck_is_empty 
# self.no_more_cards_left = no_more_cards_left
# self.button_text = button_text 
# self.mouse_clicks = 0
# self.cards_on_the_table = []  
# elf.players_cards 