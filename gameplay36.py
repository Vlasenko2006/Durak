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
game.create_deck()
player0 = attacker_net
player0.load_state_dict(torch.load("attacker1_1100"))



class CardPlotter(tk.Tk):
    def __init__(self, card_width, 
                 card_height,
                 num_closed_cards,
                 num_open_cards,
                 button_text,
                 attack_flag,
                 deck_is_empty = False, 
                 no_more_cards_left = False,
                 distance=2.5, 
                 factor=1):
        super().__init__()
        
        # Gameplay settings
        self.deck_is_empty = deck_is_empty 
        self.no_more_cards_left = no_more_cards_left
        self.button_text = button_text 
        self.mouse_clicks = 0
        self.cards_on_the_table = []  
        self.players_cards = []  
        self.num_open_cards = num_open_cards
        self.num_closed_cards = num_closed_cards
        self.attack_flag = attack_flag
        self.is_destroying = False  # Flag to indicate whether the application is being destroyed
        
        
        deck_status = game.refill_hands(attacker,defender)
        if deck_status == 0:
            if not game.players[attacker]:
               # print("Loop is done")
                big_loop_done = True
            if not game.players[defender]:
               # print("Loop is done")
                big_loop_done = True
        
        
        
        
        self.title("Durak Game")

        # Get the screen width and calculate half of it
        screen_width = self.winfo_screenwidth()
        frame_width = screen_width // 2

        # Define card dimensions and spacing
        self.factor = factor
        self.card_width = card_width
        self.card_height = card_height
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
        # This additional frame will be referred to as the table
        self.table = tk.Frame(self.frame, bg='blue', width=2 * self.card_width, height=self.card_height)
        self.table.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Store image references to prevent garbage collection
        self.image_refs = []
        self.lower_card_labels = []
        self.upper_card_labels = []
        self.table_card_labels = []  # Store references to cards placed on the table

        # Create and place closed card images at the top of the frame
        self.create_closed_cards(num_closed_cards)
        # Create and place open card images at the bottom of the frame
        self.create_open_cards(num_open_cards)
        # Draw the trump card perpendicular to the deck
        self.draw_trump_card_perpendicular()
        # Draw a card back opposite to the left side of the table
        self.draw_card_back_opposite_left()

        # Add the Finish button just above the row with open cards
        self.add_finish_button()

        # Start the game based on the attack_flag
        if self.attack_flag == -1:
            self.after(1000, self.pop_card_from_top)

    def create_card_image(self, rank, suit):
        width, height = self.card_width, self.card_height
        color = suit_colors[suit]
        suit_symbol = suit_symbols[suit]

        # Create a blank image with white background
        card_image = Image.new('RGB', (width, height), 'white')

        # Create a drawing object to draw on the image
        draw = ImageDraw.Draw(card_image)

        # Load a font
        font = ImageFont.truetype("arial.ttf", 40)

        # Draw rank in the top-left corner
        draw.text((10, 10), rank, font=font, fill=color)

        # Draw rank in the bottom-right corner
        draw.text((width - 50, height - 50), rank, font=font, fill=color)

        # Draw large suit symbol in the center (brighter)
        draw.text((width // 2 - 20, height // 2 - 40), suit_symbol, font=font, fill=color if color == 'black' else (255, 0, 0, 100))

        # Return the final image with the drawn text
        return card_image

    def create_closed_card_image(self):
        width, height = self.card_width, self.card_height

        # Create a blank image with black background
        card_image = Image.new('RGB', (width, height), 'black')
        draw = ImageDraw.Draw(card_image)

        # Draw a smiley face on the back of the card
        face_radius = 50
        face_center = (width // 2, height // 2)
        eye_radius = 5
        eye_offset_x = 20
        eye_offset_y = 20
        mouth_radius = 30
        mouth_offset_y = 15

        # Draw face
        draw.ellipse((face_center[0] - face_radius, face_center[1] - face_radius,
                      face_center[0] + face_radius, face_center[1] + face_radius), fill='yellow', outline='black')

        # Draw eyes
        draw.ellipse((face_center[0] - eye_offset_x - eye_radius, face_center[1] - eye_offset_y - eye_radius,
                      face_center[0] - eye_offset_x + eye_radius, face_center[1] - eye_offset_y + eye_radius), fill='black')
        draw.ellipse((face_center[0] + eye_offset_x - eye_radius, face_center[1] - eye_offset_y - eye_radius,
                      face_center[0] + eye_offset_x + eye_radius, face_center[1] - eye_offset_y + eye_radius), fill='black')

        # Draw mouth
        draw.arc((face_center[0] - mouth_radius, face_center[1] + mouth_offset_y - mouth_radius,
                  face_center[0] + mouth_radius, face_center[1] + mouth_offset_y + mouth_radius), start=0, end=180, fill='black')

        return card_image

    def create_closed_cards(self, num_closed_cards):
        for i in range(num_closed_cards):
            closed_card_image = self.create_closed_card_image()
            closed_card_photo = ImageTk.PhotoImage(closed_card_image)
            label = tk.Label(self.frame, image=closed_card_photo, bg='grey')
            label.image = closed_card_photo  # Store reference in label
            label.grid(row=0, column=i, padx=10, pady=10, sticky="n")
            self.image_refs.append(closed_card_photo)  # Store reference to prevent garbage collection
            self.upper_card_labels.append(label)  # Store reference to upper row card labels

        # Center the closed cards
        self.frame.grid_columnconfigure((0, num_closed_cards - 1), weight=1)

    def create_open_cards(self, num_open_cards):
        for i in range(num_open_cards):
            rank = cards[i % len(cards)][0]
            suit = cards[i % len(cards)][1]
            card_image = self.create_card_image(rank, suit)
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
        self.frame.grid_columnconfigure((0, num_open_cards - 1), weight=1)

    def on_card_click(self, event):
        # Increment the mouse click counter
        self.mouse_clicks += 1
        
        # Get the clicked label
        clicked_label = event.widget

        # Debug statement to check the clicked label
        print(f"Clicked label: {clicked_label}")

        # Ensure the clicked label exists in the list before removing it
        if clicked_label in self.lower_card_labels:
            self.lower_card_labels.remove(clicked_label)
        else:
            print("Clicked label not found in lower_card_labels list")

        # Remove the clicked card from the lower row
        clicked_label.grid_forget()

        # Calculate the new position for the card on the table
        relx_position = 0.1 + 0.05 * len(self.table_card_labels)

        # Place the clicked card in the middle of the table
        clicked_label.place(in_=self.table, relx=relx_position, rely=0.25, anchor=tk.CENTER)
        clicked_label.tkraise()  # Raise the card to the top of the stacking order
        self.table_card_labels.append(clicked_label)
        self.cards_on_the_table.append(clicked_label.card_info)

        # Wait for half a second (500 milliseconds) and then remove one black card from the top row and add it to the bottom row
        self.after(500, self.pop_card_from_top)
        
        if self.mouse_clicks == self.num_closed_cards or self.mouse_clicks == self.num_open_cards:
            self.after(2000, self.finish_game)
            return

    def pop_card_from_top(self):
        if self.is_destroying:
            return

        if self.upper_card_labels:
            # Remove the first card in the upper row
            label_to_remove = self.upper_card_labels.pop(0)
            if not self.winfo_exists():
                return
            label_to_remove.grid_forget()

            # Get the rank and suit of the card to be added to the bottom row
            rank, suit = card_top[len(card_top) - len(self.upper_card_labels) - 1]
            card_image = self.create_card_image(rank, suit)
            card_photo = ImageTk.PhotoImage(card_image)
            label = tk.Label(self.frame, image=card_photo, bg='grey')
            label.image = card_photo  # Store reference in label
            label.card_info = (rank, suit)  # Store the card information in the label

            # Calculate the new position for the card on the table
            relx_position = 0.1 + 0.05 * len(self.table_card_labels)

            # Place the card in the middle of the table
            label.place(in_=self.table, relx=relx_position, rely=0.75, anchor=tk.CENTER)
            label.tkraise()  # Raise the card to the top of the stacking order
            self.table_card_labels.append(label)
            self.cards_on_the_table.append(label.card_info)
            self.players_cards.append(label.card_info)  # Add to player0_cards

            # Check if there are no more cards left in either row
            if not self.upper_card_labels and not self.lower_card_labels:
                print("Popping card from the top")
                self.after(2000, self.finish_game)

    
    def draw_card_back_opposite_left(self):
        if not self.deck_is_empty:
            closed_card_image = self.create_closed_card_image()
            closed_card_photo = ImageTk.PhotoImage(closed_card_image)
            label = tk.Label(self.frame, image=closed_card_photo, bg='grey')
            label.image = closed_card_photo  # Store reference in label
            label.place(in_=self.table, relx=-0.9, rely=0.5, anchor=tk.CENTER)
            self.image_refs.append(closed_card_photo)  # Store reference to prevent garbage collection

    def draw_trump_card_perpendicular(self):
        if not self.no_more_cards_left:
            trump_card_image = self.create_card_image("Q", "hearts")
            trump_card_image = trump_card_image.rotate(90, expand=True)
            trump_card_photo = ImageTk.PhotoImage(trump_card_image)
            label = tk.Label(self.frame, image=trump_card_photo, bg='grey')
            label.image = trump_card_photo  # Store reference in label
            label.place(in_=self.table, relx=-0.5, rely=0.5, anchor=tk.CENTER)
            self.image_refs.append(trump_card_photo)  # Store reference to prevent garbage collection

    def add_finish_button(self):
        finish_button = tk.Button(self.frame, text=self.button_text, command=self.finish_game, font=('Helvetica', 14, 'bold'))
        finish_button.grid(row=1, columnspan=10, pady=(10, 0))

    def finish_game(self):
        self.is_destroying = True
        print(f"No more cards remain. Total mouse clicks: {self.mouse_clicks}")
        print(f"Cards on the table: {self.cards_on_the_table}")
        print(f"Player 0 cards: {self.players_cards}")  # Print player0 cards
        self.destroy()

if __name__ == "__main__":    
    # Example usage with user-specified card dimensions and number of cards
    card_width = 150  # User-specified card width
    card_height = 200  # User-specified card height
    num_closed_cards = 3  # User-specified number of closed cards
    num_open_cards = 3  # User-specified number of open cards
    
    app = CardPlotter(card_width,
                      card_height,
                      num_closed_cards,
                      num_open_cards, 
                      button_text = "Finish the attack",
                      attack_flag = 1,
                      deck_is_empty=False, 
                      factor=3
                      )
    app.mainloop()
    mouse_clicks = app.mouse_clicks
    
    app = CardPlotter(card_width, 
                      card_height,
                      num_closed_cards,
                      num_open_cards,
                      button_text = "Withdraw",
                      attack_flag = -1,
                      deck_is_empty=True, 
                      no_more_cards_left = True, 
                      factor=3)
    app.mainloop()
    mouse_clicks = app.mouse_clicks