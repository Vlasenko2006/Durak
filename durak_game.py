#!/usr/bin/env python3

import random

class DurakGame:
    def __init__(self):
        self.deck = self.create_deck()
        self.players = [[], []]  # Two players' hands
        self.deal_cards()
        self.trump = self.deck[-1]  # Last card in the shuffled deck is the trump card
        self.trump_suit = self.trump[1]  # Suit of the trump card

    def create_deck(self):
        suits = ['spades', 'hearts', 'diamonds', 'clubs']
        values = ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        deck = [(value, suit) for suit in suits for value in values]
        random.shuffle(deck)
        return deck

    def deal_cards(self):
        for _ in range(6):
            self.players[0].append(self.deck.pop())
            self.players[1].append(self.deck.pop())

    def draw_card(self, player_index):
        if self.deck:
            self.players[player_index].append(self.deck.pop())
        else:
            print(f"No more cards left in the deck to draw for player {player_index + 1}.")

    def refill_hands(self, attacker,defender):
        deck_status = 1
        for player_index in [attacker,defender]:
            while len(self.players[player_index]) < 6 and self.deck:
                self.draw_card(player_index)
            if not self.deck:
            #    print(f"Player {player_index + 1} has {len(self.players[player_index])} cards. Deck is empty, no more cards to draw.")
                deck_status = 0
        return deck_status

    def get_state(self, player_index):
        state = [0] * 36  # Initialize state with zeros
        for card in self.players[player_index]:
            index = self.card_to_index(card)
            state[index] = 1
        return state
    
    # def update_state(self, gamer_id, card_index, state):
    #     for card in self.players[gamer_id]:
    #         state[0, card_index] = state[0, card_index] + 1
    #     return state
    
    def update_state(self, gamer_id, card_index, state):
        new_state = state.clone()  # Clone the state to avoid in-place modification
        for card in self.players[gamer_id]:
            new_state[0, card_index] = 1
        return new_state

    def card_to_index(self, card):
        value, suit = card
        value_order = {'6': 0, '7': 1, '8': 2, '9': 3, '10': 4, 'J': 5, 'Q': 6, 'K': 7, 'A': 8}
        suit_order = {'spades': 0, 'hearts': 1, 'diamonds': 2, 'clubs': 3}
        return suit_order[suit] * 9 + value_order[value]

    def index_to_card(self, index):
        values = ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        suits = ['clubs', 'diamonds', 'hearts', 'spades']
        value = values[index // 4]  # Corrected value indexing
        suit = suits[index % 4]     # Corrected suit indexing
        return (value, suit)

    def can_beat(self, attack_card, defend_card):
        attack_value, attack_suit = attack_card
        defend_value, defend_suit = defend_card
        value_order = {'6': 0, '7': 1, '8': 2, '9': 3, '10': 4, 'J': 5, 'Q': 6, 'K': 7, 'A': 8}
        if defend_suit == attack_suit and value_order[defend_value] > value_order[attack_value]:
            return True
        if defend_suit == self.trump_suit and attack_suit != self.trump_suit:
            return True
        return False