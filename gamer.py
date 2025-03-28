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
from defence import defence




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
        self.players_decision = 0
        self.continue_attack  = False
 
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
    
    def opponent_attacks(self):
        
        attack_value = None # attack_value is a card (suit, value) that attacker (NN )chooses to attack
        attack_flag = torch.tensor([1.], dtype=torch.float32, requires_grad=False).unsqueeze(0)
        
        self.players_decision, _, chosen_attackers_card, attacker_card_index, \
                self.cards_on_a_table, done = attack([self.player0], 
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
        if not done: done = self.decision_to_continue_attack()
            
        return  chosen_attackers_card, attacker_card_index, done 
            
    def decision_to_continue_attack(self, verbose = False):
         # Attacker decides whether to continue attack
         done = False
         self.continue_attack = self.fplayers_decision > self.margin_attacker  # Randomly decide to continue or stop
         if not self.continue_attack:
             self.played_cards = self.not_playing_cards.clone()
             done = True
             if verbose: print("Attacker decides to stop the attack")
         return done
     
    def opponent_defends(self,chosen_attackers_card, attacker_card_index):
        defend_flag = torch.tensor([-1.], dtype=torch.float32, requires_grad=False).unsqueeze(0)
        done = False
        
        self.cards_on_a_table, _, self.defence_decision, \
        chosen_defender_card, output_defender = defence(self.game,
                     [self.player0],
                     1,
                     0 ,
                     defend_flag,
                     self.played_cards,
                     self.cards_on_a_table,
                     chosen_attackers_card,
                     attacker_card_index,
                     self.margin_defender,
                     self.deck
                     )
        if self.defence_decision == "withdraw": done = True
        
        return chosen_defender_card, done




 # Convert to Float










        
        # Gameplay settings
# self.deck_is_empty = deck_is_empty 
# self.no_more_cards_left = no_more_cards_left
# self.button_text = button_text 
# self.mouse_clicks = 0
# self.cards_on_the_table = []  
# elf.players_cards 