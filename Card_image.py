import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont

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

class Card_image:
    def __init__(self, 
                 card_width=150,
                 card_height=200,
                 distance =2.5):
        self.card_width = card_width
        self.card_height = card_height
        self.row_spacing = int(distance * self.card_height)
        self.frame_height = self.card_height * 2 + self.row_spacing + 20

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

    def create_closed_cards(self, opponent_cards,image_refs, upper_card_labels,frame):
        upper_card_labels.clear()
        for i in range(len(opponent_cards)):
            closed_card_image = self.create_closed_card_image()
            closed_card_photo = ImageTk.PhotoImage(closed_card_image)
            label = tk.Label(frame, image=closed_card_photo, bg='grey')
            label.image = closed_card_photo  # Store reference in label
            label.grid(row=0, column=i, padx=10, pady=10, sticky="n")
            image_refs.append(closed_card_photo)  # Store reference to prevent garbage collection
            upper_card_labels.append(label)  # Store reference to upper row card labels

        # Center the closed cards
        frame.grid_columnconfigure((0, len(opponent_cards) - 1), weight=1)



    def create_open_cards(self,my_cards, image_refs,lower_card_labels, on_card_click, frame):
        lower_card_labels.clear()
        for i in range(len(my_cards)):
            rank = my_cards[i][0]
            suit = my_cards[i][1]
            card_image = self.create_card_image(rank, suit)
            card_photo = ImageTk.PhotoImage(card_image)
            label = tk.Label(frame, image=card_photo, bg='grey')
            label.image = card_photo  # Store reference in label
            label.grid(row=2, column=i, padx=10, pady=(self.row_spacing, 10), sticky="n")
            label.bind("<Button-1>", on_card_click)  # Bind left mouse button click event
            image_refs.append(card_photo)  # Store reference to prevent garbage collection
            lower_card_labels.append(label)  # Store reference to lower row card labels
           # Store the card information in the label
            label.card_info = (rank, suit)


        
    def draw_trump_card_perpendicular(self, 
                                          trump,
                                          no_more_cards_left,   
                                          image_refs,
                                          table,
                                          frame):
        if not no_more_cards_left:
            trump_card_image = self.create_card_image(trump[0] ,trump[1])
            trump_card_image = trump_card_image.rotate(90, expand=True)
            trump_card_photo = ImageTk.PhotoImage(trump_card_image)
            label = tk.Label(frame, image=trump_card_photo, bg='grey')
            label.image = trump_card_photo
            label.place(in_= table, relx=-0.5, rely=0.5, anchor=tk.CENTER)
            image_refs.append(trump_card_photo)