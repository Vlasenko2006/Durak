import random

class DurakGame:
    def __init__(self):
        self.deck = self.create_deck()
        self.players = [[], []]  # Two players' hands
        self.deal_cards()
        self.trump = self.deck[-1]  # Last card in the shuffled deck is the trump card
        self.trump_suit = self.trump[1]  # Suit of the trump card

    def get_deck(self):
        suits = ['spades', 'hearts', 'diamonds', 'clubs']
        values = ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        return [(value, suit) for suit in suits for value in values]

    def create_deck(self):
        deck = self.get_deck()
        random.shuffle(deck)
        return deck

    def deal_cards(self):
        for _ in range(6):
            self.players[0].append(self.deck.pop(0))
            self.players[1].append(self.deck.pop(0))

    def draw_card(self, player_index):
        if self.deck:
            self.players[player_index].append(self.deck.pop(0))
        else:
             print(f"No more cards left in the deck to draw for player {player_index + 1}.")

    def refill_hands(self, attacker, defender):
        deck_status = 1
        for player_index in [attacker, defender]:
            while len(self.players[player_index]) < 6 and self.deck:
                self.draw_card(player_index)
            if not self.deck:
                deck_status = 0
    #    print(f"Debug: Remaining cards in the deck after refill_hands: {len(self.deck)}")
        return deck_status

    def get_state(self, player_index):
        state = [0] * 36  # Initialize state with zeros
        for card in self.players[player_index]:
            index = self.card_to_index(card)
            state[index] = 1
        return state

    def update_state(self, gamer_id, card_index, state):
        # Clone the state to avoid in-place modification
        new_state = state.clone()
        
        # Update the state based on the defender's action
        new_state[0, card_index] = 1
        
        # print(f"Debug: Updating state for gamer_id {gamer_id} with card_index {card_index}")
        # print(f"Debug: Updated state: {new_state}")
        
        return new_state
    
    def card_to_index(self, card):
        value, suit = card
        deck = self.get_deck()
        values = [v[0] for v in deck[:9]]  # Extract values from the first 9 cards (one suit)
        suits = [s[1] for s in deck[::9]]  # Extract suits from every 9th card
        value_order = {v: i for i, v in enumerate(values)}
        suit_order = {s: i for i, s in enumerate(suits)}
        index = suit_order[suit] * 9 + value_order[value]
        # print(f"Debug: card_to_index: card = {card}, index = {index}")
        return index

    def index_to_card(self, index):
        deck = self.get_deck()
        values = [v[0] for v in deck[:9]]  # Extract values from the first 9 cards (one suit)
        suits = [s[1] for s in deck[::9]]  # Extract suits from every 9th card
        value = values[index % 9]  # Corrected value indexing
        suit = suits[index // 9]   # Corrected suit indexing
        card = (value, suit)
        # print(f"Debug: index_to_card: index = {index}, card = {card}")
        return card

    def can_beat(self, attack_card, defend_card):
        attack_value, attack_suit = attack_card
        defend_value, defend_suit = defend_card
        deck = self.get_deck()
        values = [v[0] for v in deck[:9]]  # Extract values from the first 9 cards (one suit)
        value_order = {v: i for i, v in enumerate(values)}
        if defend_suit == attack_suit and value_order[defend_value] > value_order[attack_value]:
            return True
        if defend_suit == self.trump_suit and attack_suit != self.trump_suit:
            return True
        return False

    def indexes_to_cards(self, card_indexes):
        cards = []
        for i, j in enumerate(card_indexes.squeeze()):
            if int(j) == 1:  # Process all indexes where the value is 1
                cards.append(self.index_to_card(i))
                # print("Debug: cards", cards)
        return cards