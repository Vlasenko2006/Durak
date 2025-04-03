import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
from gamer import gamer
from Card_image import Card_image

gamer_instance = gamer()
card_im = Card_image()

class CardPlotter(tk.Tk):
    def __init__(self, deck, trump, button_text, attack_flag, deck_is_empty=False, no_more_cards_left=False, distance=2.5, factor=1):
        super().__init__()

        # Gameplay settings
        self.gamer = gamer_instance
        self.deck = deck
        self.deck_is_empty = deck_is_empty 
        self.no_more_cards_left = no_more_cards_left
        self.button_text = button_text 
        self.mouse_clicks = 0
        self.cards_on_the_table = []  
        self.players_cards = []  
        self.attack_flag = attack_flag
        self.is_destroying = False  # Flag to indicate whether the application is being destroyed

        self.opponent_cards = self.gamer.game.players[0]        
        self.my_cards = self.gamer.game.players[1]
        self.num_closed_cards = len(self.opponent_cards)
        self.num_open_cards = len(self.my_cards)
        self.my_card = None
        self.opponents_card = None
        self.opponent_card_index = None
        self.trump = trump
        
        self.done = False

        # Store image references to prevent garbage collection
        self.image_refs = []
        self.lower_card_labels = []
        self.upper_card_labels = []
        self.table_card_labels = []  # Store references to cards placed on the table
        
        self.title("Durak Game")

        # Get the screen width and calculate half of it
        screen_width = self.winfo_screenwidth()
        frame_width = screen_width // 2

        # Define card dimensions and spacing
        self.factor = factor
        self.card_width = card_im.card_width
        self.card_height = card_im.card_height
        self.row_spacing = int(distance * self.card_height)

        # Calculate frame dimensions
        frame_height = self.card_height * 2 + self.row_spacing + 20

        # Create a larger frame for the cards
        self.frame = tk.Frame(self, bg='grey', width=frame_width, height=frame_height)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Fix the frame size
        self.frame.pack_propagate(False)

        # Set minimum size for the window to prevent shrinking
        self.minsize(frame_width, frame_height)

        # Create the additional frame (table) with the same height and width equal to the width of two cards
        self.table = tk.Frame(self.frame, bg='blue', width=2 * self.card_width, height=self.card_height)
        self.table.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Create and place closed card images at the top of the frame
        self.create_closed_cards()
        # Create and place open card images at the bottom of the frame
        self.create_open_cards()
        # Draw the trump card perpendicular to the deck
        self.draw_trump_card_perpendicular()
        # Draw a card back opposite to the left side of the table
        self.draw_card_back_opposite_left()

        # Add the Finish button just above the row with open cards
        self.add_finish_button()

        # Start the game based on the attack_flag
        self.start_game()

    def start_game(self):
        if self.attack_flag == 1:
            self.after(1000, self.perform_attack)
        elif self.attack_flag == -1:
            self.enable_bottom_cards()

    def enable_bottom_cards(self):
        for label in self.lower_card_labels:
            label.bind("<Button-1>", self.on_card_click)

    def create_closed_cards(self):
        self.upper_card_labels.clear()
        for i in range(len(self.opponent_cards)):
            closed_card_image = card_im.create_closed_card_image()
            closed_card_photo = ImageTk.PhotoImage(closed_card_image)
            label = tk.Label(self.frame, image=closed_card_photo, bg='grey')
            label.image = closed_card_photo  # Store reference in label
            label.grid(row=0, column=i, padx=10, pady=10, sticky="n")
            self.image_refs.append(closed_card_photo)  # Store reference to prevent garbage collection
            self.upper_card_labels.append(label)  # Store reference to upper row card labels

        # Center the closed cards
        self.frame.grid_columnconfigure((0, len(self.opponent_cards) - 1), weight=1)

    def create_open_cards(self):
        self.lower_card_labels.clear()
        for i in range(len(self.my_cards)):
            rank = self.my_cards[i][0]
            suit = self.my_cards[i][1]
            card_image = card_im.create_card_image(rank, suit)
            card_photo = ImageTk.PhotoImage(card_image)
            label = tk.Label(self.frame, image=card_photo, bg='grey')
            label.image = card_photo  # Store reference in label
            label.grid(row=2, column=i, padx=10, pady=(self.row_spacing, 10), sticky="n")
            label.bind("<Button-1>", self.on_card_click)  # Bind left mouse button click event
            self.image_refs.append(card_photo)  # Store reference to prevent garbage collection
            self.lower_card_labels.append(label)  # Store reference to lower row card labels
            # Store the card information in the label
            label.card_info = (rank, suit)

        # Center the open cards
        self.frame.grid_columnconfigure((0, len(self.my_cards) - 1), weight=1)

    def update_opponent_cards(self):
        # Remove the first card from the upper row and update the upper row
        if self.upper_card_labels:
            label_to_remove = self.upper_card_labels.pop(0)
            label_to_remove.grid_forget()

    def perform_attack(self):
        if self.upper_card_labels:
           # print("UPPER CARD LABELS = ", self.upper_card_labels)
            self.mouse_clicks += 1
            attack_card, attacker_card_index, done = self.gamer.opponent_attacks()  # Updated to handle three returned values
            if attack_card is None:
                return
            rank, suit = attack_card
            card_image = card_im.create_card_image(rank, suit)
            card_photo = ImageTk.PhotoImage(card_image)
            label = tk.Label(self.frame, image=card_photo, bg='grey')
            label.image = card_photo
            label.card_info = (rank, suit)

            relx_position = 0.1 + 0.05 * len(self.table_card_labels)
            label.place(in_=self.table, relx=relx_position, rely=0.25, anchor=tk.CENTER)
            label.tkraise()
            self.table_card_labels.append(label)
            self.cards_on_the_table.append(label.card_info)
            self.players_cards.append(label.card_info)

            # Update opponent cards after attack
            self.update_opponent_cards()

            if self.mouse_clicks != self.num_closed_cards and self.mouse_clicks != self.num_open_cards:
                self.after(500, self.enable_bottom_cards)
            else:
                self.after(2000, self.finish_game)

    def on_card_click(self, event):
        clicked_label = event.widget
        self.my_card = clicked_label.card_info
      #  print(f"Clicked label: {clicked_label}, Card info: {self.my_card}")

        if clicked_label in self.lower_card_labels:
            self.lower_card_labels.remove(clicked_label)
        else:
            print("Clicked label not found in lower_card_labels list")

        # Remove the card from my_cards
        self.my_cards.remove(self.my_card)

        clicked_label.grid_forget()
        relx_position = 0.1 + 0.05 * len(self.table_card_labels)
        clicked_label.place(in_=self.table, relx=relx_position, rely=0.25, anchor=tk.CENTER)
        clicked_label.tkraise()
        self.table_card_labels.append(clicked_label)
        self.cards_on_the_table.append(clicked_label.card_info)

        if  attack_flag ==1:
            self.after(500, self.perform_attack)
        else:
            self.after(500, self.perform_defense)

    def perform_defense(self):
        defense_card, self.defence_decision, done = self.gamer.opponent_defends(self.my_card, self.gamer.game.card_to_index(self.my_card))
        if defense_card is None:
            self.finish_game(self.defence_decision)
            return

        rank, suit = defense_card
        card_image = card_im.create_card_image(rank, suit)
        card_photo = ImageTk.PhotoImage(card_image)
        label = tk.Label(self.frame, image=card_photo, bg='grey')
        label.image = card_photo
        label.card_info = (rank, suit)

        relx_position = 0.1 + 0.05 * len(self.table_card_labels)
        label.place(in_=self.table, relx=relx_position, rely=0.75, anchor=tk.CENTER)
        label.tkraise()
        self.table_card_labels.append(label)
        self.cards_on_the_table.append(label.card_info)
        self.players_cards.append(label.card_info)

        # Update opponent cards after defense
        self.update_opponent_cards()

        if not self.upper_card_labels and not self.lower_card_labels:
            print("No more cards left")
            self.after(2000, self.finish_game)

    def draw_card_back_opposite_left(self):
        if not self.deck_is_empty:
            closed_card_image = card_im.create_closed_card_image()
            closed_card_photo = ImageTk.PhotoImage(closed_card_image)
            label = tk.Label(self.frame, image=closed_card_photo, bg='grey')
            label.image = closed_card_photo
            label.place(in_=self.table, relx=-0.9, rely=0.5, anchor=tk.CENTER)
            self.image_refs.append(closed_card_photo)

    def draw_trump_card_perpendicular(self):
        if not self.no_more_cards_left:
            trump_card_image = card_im.create_card_image(self.trump[0] ,self.trump[1])
            trump_card_image = trump_card_image.rotate(90, expand=True)
            trump_card_photo = ImageTk.PhotoImage(trump_card_image)
            label = tk.Label(self.frame, image=trump_card_photo, bg='grey')
            label.image = trump_card_photo
            label.place(in_=self.table, relx=-0.5, rely=0.5, anchor=tk.CENTER)
            self.image_refs.append(trump_card_photo)

    def add_finish_button(self):
        finish_button = tk.Button(self.frame, text=self.button_text, command=self.finish_game, font=('Helvetica', 14, 'bold'))
        finish_button.grid(row=1, columnspan=10, pady=(10, 0))

    def finish_game(self, message="No message"):
        print(message)
        self.destroy()
        return self.cards_on_the_table

    # def refill_hands(self):
    #     self.gamer.refill()

if __name__ == "__main__":    
    card_width = 150  # User-specified card width
    card_height = 200  # User-specified card height

    done = False
    deck_is_empty = False
    no_more_cards_left = False
    deck = gamer_instance.game.deck
    trump = deck[-1]
    attacker = 0
    defender = 1
    attack_flag = 1
    counter = 0
    while not done:
        print("Deck before", deck)
        counter +=1
        app = CardPlotter(deck, trump, button_text="Finish", attack_flag= attack_flag, deck_is_empty=deck_is_empty, no_more_cards_left = deck_is_empty, factor=3)
        app.mainloop()
        # print("app.cards_on_the_table = ", app.cards_on_the_table)
        deck_status = gamer_instance.game.refill_hands(attacker, defender)
        print("======================== \n")
        print("TRUMP = ", trump)
        print("Deck after", deck)
        print("app.my_cards = ", app.my_cards)
        print("app.opponent_cards = ", app.opponent_cards)   
  
        
    
    
        attack_flag= -1 * attack_flag
        if deck_status == 0:
            deck_is_empty = True
            if not app.opponent_cards or not app.my_cards:
                done = True
            
        print("len of the deck = ",len(deck))
        if counter == 4: done = True
                
                
        
        #print("Cards on the table:", app.cards_on_the_table)