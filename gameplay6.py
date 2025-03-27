import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont

# Define the cards
cards = [
    ("5", "clubs"),
    ("J", "diamonds"),
    ("A", "hearts"),
    ("Q", "spades"),
    ("K", "hearts")
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
    def __init__(self, card_width, card_height, distance=2.5):
        super().__init__()
        self.title("Card Plotter")

        # Define card dimensions and spacing
        self.card_width = card_width
        self.card_height = card_height
        self.row_spacing = int(distance * self.card_height)

        # Calculate frame dimensions
        frame_width = self.card_width * 5 + 20
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
        self.create_closed_cards()
        # Create and place open card images at the bottom of the frame
        self.create_open_cards()

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

    def create_empty_frame(self):
        width, height = self.card_width, self.card_height

        # Create a blank image with transparent background
        empty_frame_image = Image.new('RGB', (width, height), 'grey')
        draw = ImageDraw.Draw(empty_frame_image)

        # Draw a black rectangle frame
        draw.rectangle([(0, 0), (width-1, height-1)], outline='black', width=3)

        return empty_frame_image

    def create_closed_cards(self):
        for i in range(5):
            closed_card_image = self.create_closed_card_image()
            closed_card_photo = ImageTk.PhotoImage(closed_card_image)
            label = tk.Label(self.frame, image=closed_card_photo, bg='grey')
            label.image = closed_card_photo  # Store reference in label
            label.grid(row=0, column=i, padx=10, pady=10)
            self.image_refs.append(closed_card_photo)  # Store reference to prevent garbage collection
            self.upper_card_labels.append(label)  # Store reference to upper row card labels

    def create_open_cards(self):
        for i, (rank, suit) in enumerate(cards):
            card_image = self.create_card_image(rank, suit)
            card_photo = ImageTk.PhotoImage(card_image)
            label = tk.Label(self.frame, image=card_photo, bg='grey')
            label.image = card_photo  # Store reference in label
            label.grid(row=1, column=i, padx=10, pady=(self.row_spacing, 10))
            label.bind("<Button-1>", self.on_card_click)  # Bind left mouse button click event
            self.image_refs.append(card_photo)  # Store reference to prevent garbage collection
            self.lower_card_labels.append(label)  # Store reference to lower row card labels

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
        pady_upwards_shift = int(self.card_height * 0.75)  # Adjust this value to shift more upwards

        # Place the clicked card in the middle of the frame and shifted upwards
        clicked_label.grid(row=middle_row, column=middle_column, padx=10, pady=(self.row_spacing - pady_upwards_shift, 10))

        # Wait for half a second (500 milliseconds) and then substitute one black card from the top row with an empty frame
        self.after(500, self.substitute_black_card_with_frame)

    def substitute_black_card_with_frame(self):
        if self.upper_card_labels:
            # Remove the first card in the upper row and substitute with an empty frame
            label_to_substitute = self.upper_card_labels.pop(0)
            label_to_substitute.grid_forget()

            empty_frame_image = self.create_empty_frame()
            empty_frame_photo = ImageTk.PhotoImage(empty_frame_image)
            label = tk.Label(self.frame, image=empty_frame_photo, bg='grey')
            label.image = empty_frame_photo  # Store reference in label
            label.grid(row=0, column=len(self.upper_card_labels), padx=10, pady=10)
            self.image_refs.append(empty_frame_photo)  # Store reference to prevent garbage collection

if __name__ == "__main__":
    # Example usage with user-specified card dimensions
    card_width = 150  # User-specified card width
    card_height = 200  # User-specified card height
    app = CardPlotter(card_width, card_height)
    app.mainloop()