#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 13:16:17 2025

@author: andrey
"""

import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
from neural_networks import CardNN
import torch
from durak_game import DurakGame
from attack import attack




class gamer:
    def __init__(self):
        self.game = DurakGame()
        self.deck = self.game.create_deck()
        self.game.deal_cards()
        self.trump = self.deck[-1]  # Last card in the shuffled deck is the trump card
        self.trump_suit = self.trump[1]  # Suit of the trump card
        self.player0 =  CardNN()
        self.load_player()
        self.played_cards = torch.zeros(1,36)
        self.taken_cards = torch.zeros(2,36)
        self.cards_on_a_table = torch.zeros(1, 36, dtype=torch.float32) 
        self.opponents_cards = torch.zeros(1, 36, dtype=torch.float32) 
        self.my_cards = torch.zeros(1, 36, dtype=torch.float32) 
        self.opponents_cards = self.game.players[0]
        self.my_cards = self.game.players[1]
        self.not_playing_cards = self.played_cards.clone().float()  # Convert to Float to avoid type mismatch
        self.margin_attacker = 0.5
        self.margin_defender = 0.5 
        
 
    def load_player(self, checkpoint = "attacker1_1100"):
        self.player0.load_state_dict(torch.load(checkpoint))
        
    def refill(self):
        deck_status = self.game.refill_hands(self.opponents_cards,self.my_cards)
        winner = "no winner yet"
        if deck_status == 0:
            if not self.game.players[0]:
                winner = "You Lost"
            if not self.game.players[1]:
                winner = "You Win"
        return winner
    
    def opponent_attack(self):
        
        attack_value = None # attack_value is a card (suit, value) that attacker (NN )chooses to attack
        attack_flag = torch.tensor([1.], dtype=torch.float32, requires_grad=True).unsqueeze(0)
        
        decision_to_continue_attack,attacker_card_prob,\
            chosen_attackers_card, attacker_card_index, \
                self.cards_on_a_table, done, \
                    output_attacker = attack([self.player0], 
                           0,
                           attack_value, 
                           self.game,
                           attack_flag,
                           self.played_cards,
                           self.cards_on_a_table,
                           self.deck, 
                           1, # episode = 1 
                           verbose = False
                           )
        return players_decision, attacker_card_prob, chosen_attackers_card, \
            attacker_card_index, done, output_attacker
            
    def decision_to_continue_attack(self,players_decision):
         # Attacker decides whether to continue attack
         continue_attack = players_decision > self.margin_attacker  # Randomly decide to continue or stop
         if not continue_attack:
             self.played_cards = self.not_playing_cards.clone()
             done = True
             if verbose: print(f"Episode {episode + 1}: Attacker decides to stop the attack.")
             break
    return done




 # Convert to Float










        
        # Gameplay settings
# self.deck_is_empty = deck_is_empty 
# self.no_more_cards_left = no_more_cards_left
# self.button_text = button_text 
# self.mouse_clicks = 0
# self.cards_on_the_table = []  
# elf.players_cards 