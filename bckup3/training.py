import torch
import torch.nn.functional as F
from durak_game import DurakGame
from neural_networks import attacker_net, defender_net, attacker_optimizer, defender_optimizer
import matplotlib.pyplot as plt

# Hyperparameters
gamma = 0.99
batch_size = 64
num_episodes = 100
num_games_to_visualize = 3

# Unicode characters for card suits
suit_symbols = {
    'clubs': '♣',
    'diamonds': '♦',
    'hearts': '♥',
    'spades': '♠'
}

# Store game data for visualization
game_data = []

def mask_invalid_cards(action_probs, valid_cards, deck):
    # Create a mask for valid cards
    mask = torch.zeros_like(action_probs)
    for card in valid_cards:
        index = deck.index(card)
        mask[0, index] = 1
    # Apply mask to action probabilities
    masked_probs = action_probs * mask
    # Normalize masked probabilities to sum to 1
    if masked_probs.sum().item() != 0:
        masked_probs = masked_probs / masked_probs.sum()
    return masked_probs, mask

def train_step():
    # Placeholder for training logic
    pass

def train_networks():
    # Define the deck of cards
    deck = [(rank, suit) for rank in ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] for suit in ['clubs', 'diamonds', 'hearts', 'spades']]
    
    game = None  # Initialize game to None

    for episode in range(num_episodes):
        game = DurakGame()
        state_attacker = torch.tensor(game.get_state(0), dtype=torch.float32, requires_grad=True).unsqueeze(0)
        state_defender = torch.tensor(game.get_state(1), dtype=torch.float32, requires_grad=True).unsqueeze(0)
        done = False

        # Store the cards for visualization
        attack_cards = []
        defense_cards = []

        defender_action_probs = None  # Initialize defender_action_probs

        step_number = 0  # Initialize step number
        attack_value = None  # Initialize attack value

        # Initialize rewards
        reward_attacker = 0
        reward_defender = 0

        while not done:
            step_number += 1  # Increment step number

            # Attacker's turn
            valid_attacker_cards = [card for card in game.players[0] if card in deck]
            if attack_value:
                valid_attacker_cards = [card for card in valid_attacker_cards if card[0] == attack_value]
            
            if not valid_attacker_cards:
                done = True
                print(f"Episode {episode + 1}: No valid cards to attack.")
                break
            
            attacker_action_probs = attacker_net(state_attacker)
            masked_attacker_action_probs, mask = mask_invalid_cards(attacker_action_probs, valid_attacker_cards, deck)
            attacker_card_index = torch.argmax(masked_attacker_action_probs).item()
            chosen_card = game.index_to_card(attacker_card_index)

            if chosen_card not in game.players[0]:
                # Invalid move, attacker loses
                reward_attacker = -100
                reward_defender = 100
                done = True
                defender_action_probs = torch.zeros_like(attacker_action_probs, requires_grad=True)  # Initialize defender_action_probs to avoid unbound error
                print(f"Episode {episode + 1}: Invalid move by attacker.")
                return  # Stop the training loop if an invalid card is chosen
            else:
                # Valid move, proceed with defense
                attack_cards.append((chosen_card, step_number))
                game.players[0].remove(chosen_card)
                state_defender = torch.tensor(game.get_state(1), dtype=torch.float32, requires_grad=True).unsqueeze(0)

                if attack_value is None:
                    attack_value = chosen_card[0]

                # Defender's turn
                print(f"Episode {episode + 1}: Defender's turn.")
                defender_action_probs = defender_net(state_defender)
                masked_defender_action_probs, _ = mask_invalid_cards(defender_action_probs, game.players[1], deck)
                defender_card_index = torch.argmax(masked_defender_action_probs).item()
                chosen_defender_card = game.index_to_card(defender_card_index)

                if chosen_defender_card not in game.players[1] or not game.can_beat(chosen_card, chosen_defender_card):
                    # Invalid move or cannot beat, defender loses
                    reward_attacker = 100
                    reward_defender = -100
                    done = True
                    print(f"Episode {episode + 1}: Attacker wins.")
                else:
                    # Valid defense, proceed with next round
                    defense_cards.append(chosen_defender_card)
                    print(f"Episode {episode + 1}: Defense Card: {chosen_defender_card}")
                    game.players[1].remove(chosen_defender_card)
                    state_attacker = torch.tensor(game.get_state(0), dtype=torch.float32, requires_grad=True).unsqueeze(0)

                    # Check win conditions and rewards
                    if not game.players[0]:
                        reward_attacker = 100
                        reward_defender = -100
                        done = True
                        print(f"Episode {episode + 1}: Attacker wins.")
                    elif not game.players[1]:
                        reward_attacker = -100
                        reward_defender = 100
                        done = True
                        print(f"Episode {episode + 1}: Defender wins.")

                if not any(card[0] == attack_value for card in game.players[0]):
                    attack_value = None
                    done = True
                    print(f"Episode {episode + 1}: Attacker has no more cards of the same value.")
                    break

                # Attacker decides whether to continue attack
                continue_attack = torch.rand(1).item() > 0.05  # Randomly decide to continue or stop
                if not continue_attack:
                    done = True
                    print(f"Episode {episode + 1}: Attacker decides to stop the attack.")
                    break

        # Store game data for visualization
        game_data.append((attack_cards, defense_cards, game.trump_suit, step_number))
        if len(game_data) > num_games_to_visualize:
            game_data.pop(0)

        # Ensure attacker_action and defender_action have consistent sizes
        target_attacker = torch.full_like(attacker_action_probs, reward_attacker, dtype=torch.float32)
        target_defender = torch.full_like(defender_action_probs, reward_defender, dtype=torch.float32)

        # Update networks with rewards
        attacker_optimizer.zero_grad()
        defender_optimizer.zero_grad()

        loss_attacker = F.mse_loss(attacker_action_probs, target_attacker)
        loss_defender = F.mse_loss(defender_action_probs, target_defender)

        loss_attacker.backward(retain_graph=True)
        loss_defender.backward()

        attacker_optimizer.step()
        defender_optimizer.step()

        print(f"Episode: {episode + 1}, Reward Attacker: {reward_attacker}, Reward Defender: {reward_defender}")

    # Visualize the last 3 games
    print("game_data =", game_data)
    visualize_games(game_data)

def visualize_games(game_data):
    fig, axes = plt.subplots(len(game_data), 1, figsize=(10, 5 * len(game_data)))

    for i, (attack_cards, defense_cards, trump_suit, step_number) in enumerate(game_data):
        ax = axes[i] if len(game_data) > 1 else axes
        ax.set_title(f"Game {i + 1} (Trump: {suit_symbols[trump_suit]})")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1, max(len(attack_cards), len(defense_cards)))
        ax.set_ylim(-1, 2)

        # Plot attacking cards
        for j, (card, step) in enumerate(attack_cards):
            card_text = f"{card[0]} {suit_symbols[card[1]]}"
            ax.text(j, 1, card_text, ha='center', va='center', fontsize=12, color='red')
            ax.text(j, 1.2, suit_symbols[card[1]], ha='left', va='top', fontsize=10, color='black')
            ax.text(j, 0.8, suit_symbols[card[1]], ha='right', va='bottom', fontsize=10, color='black')
            ax.text(j, 1.4, f"Step {step}", ha='center', va='center', fontsize=10, color='black')

        # Plot defending cards
        for j, card in enumerate(defense_cards):
            card_text = f"{card[0]} {suit_symbols[card[1]]}"
            ax.text(j, 0, card_text, ha='center', va='center', fontsize=12, color='blue')
            ax.text(j, 0.2, suit_symbols[card[1]], ha='left', va='top', fontsize=10, color='black')
            ax.text(j, -0.2, suit_symbols[card[1]], ha='right', va='bottom', fontsize=10, color='black')

        # Add game result if the game is over
        if attack_cards and attack_cards[-1][1] == step_number:
            ax.text(0.5, 1.5, "Attacker Wins", ha='center', va='center', fontsize=16, color='green')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_networks()