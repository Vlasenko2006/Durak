import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont

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

class CardPlotter(tk.Tk):
    def __init__(self, card_width, card_height, num_closed_cards, num_open_cards, distance=2.5, factor = 1):
        super().__init__()
        self.title("Card Plotter")

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

        # Store image references to prevent garbage collection
        self.image_refs = []
        self.lower_card_labels = []
        self.upper_card_labels = []

        # Create and place closed card images at the top of the frame
        self.create_closed_cards(num_closed_cards)
        # Create and place open card images at the bottom of the frame
        self.create_open_cards(num_open_cards)

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
        mouth_offset_y = 20

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
            label.grid(row=1, column=i, padx=10, pady=(self.row_spacing, 10), sticky="n")
            label.bind("<Button-1>", self.on_card_click)  # Bind left mouse button click event
            self.image_refs.append(card_photo)  # Store reference to prevent garbage collection
            self.lower_card_labels.append(label)  # Store reference to lower row card labels

        # Center the open cards
        self.frame.grid_columnconfigure((0, num_open_cards - 1), weight=1)

    def on_card_click(self, event):
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

        # Calculate middle position and shift upwards
        middle_row = 1
        middle_column = 2
        pady_upwards_shift = int(self.card_height * (self.factor -.75))  # Adjust this value to shift more upwards

        # Place the clicked card in the middle of the frame and shifted upwards
        clicked_label.grid(row=middle_row, column=middle_column, padx=10, pady=(self.row_spacing - pady_upwards_shift, 10))

        # Wait for half a second (500 milliseconds) and then remove one black card from the top row and add it to the bottom row
        self.after(500, self.pop_card_from_top)

    def pop_card_from_top(self):
        if self.upper_card_labels:
            # Remove the first card in the upper row
            label_to_remove = self.upper_card_labels.pop(0)
            label_to_remove.grid_forget()

            # Get the rank and suit of the card to be added to the bottom row
            rank, suit = card_top[len(card_top) - len(self.upper_card_labels) - 1]
            card_image = self.create_card_image(rank, suit)
            card_photo = ImageTk.PhotoImage(card_image)
            label = tk.Label(self.frame, image=card_photo, bg='grey')
            label.image = card_photo  # Store reference in label

            # Place the card in the middle of the frame, below the last clicked card
            middle_row = 1
            middle_column = 2
            pady_downwards_shift = int(self.card_height * (self.factor -.5))  # Adjust this value to shift more downwards
            label.grid(row=middle_row, column=middle_column, padx=10, pady=(self.row_spacing - pady_downwards_shift, 10))

            # Re-center the upper row and lower row after removing and adding a card     # FIXME. Needs centering
            # self.center_card_row(len(self.upper_card_labels), row=0)
            # self.center_card_row(len(self.lower_card_labels), row=1)

if __name__ == "__main__":
    # Example usage with user-specified card dimensions and number of cards
    card_width = 150  # User-specified card width
    card_height = 200  # User-specified card height
    num_closed_cards = 7  # User-specified number of closed cards
    num_open_cards = 4  # User-specified number of open cards
    app = CardPlotter(card_width, card_height, num_closed_cards, num_open_cards, factor = 3)
    app.mainloop()