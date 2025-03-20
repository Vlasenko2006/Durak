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
    
    for episode in range(num_episodes):
        game = DurakGame()
        state_attacker = torch.tensor(game.get_state(0), dtype=torch.float32, requires_grad=True).unsqueeze(0)
        state_defender = torch.tensor(game.get_state(1), dtype=torch.float32, requires_grad=True).unsqueeze(0)
        done = False

        # Store the cards for visualization
        attack_cards = []
        defense_cards = []

        defender_action_probs = None  # Initialize defender_action_probs

        while not done:
            # Attacker's turn
          #  print(f"Episode {episode + 1}: Attacker's turn.")
          #  print(f"Attacker's cards: {game.players[0]}")
            valid_attacker_cards = [card for card in game.players[0] if card in deck]
          #  print(f"Valid attack cards: {valid_attacker_cards}")
            attacker_action_probs = attacker_net(state_attacker)
            masked_attacker_action_probs, mask = mask_invalid_cards(attacker_action_probs, game.players[0], deck)
            attacker_card_index = torch.argmax(masked_attacker_action_probs).item()
            chosen_card = game.index_to_card(attacker_card_index)
            
            # Print IDs with card names
            # print(f"Deck IDs and Cards: {[(i, card) for i, card in enumerate(deck)]}")
            # print(f"Attacker's card IDs: {[deck.index(card) for card in game.players[0]]}")
            # print(f"Mask IDs: {[i for i, m in enumerate(mask[0]) if m == 1]}")
            # print(f"Chosen card ID: {attacker_card_index}")
            # print(f"Chosen card: {chosen_card}")

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
                attack_cards.append(chosen_card)
#                print(f"Episode {episode + 1}: Attack Card: {chosen_card}")
                game.players[0].remove(chosen_card)
                state_defender = torch.tensor(game.get_state(1), dtype=torch.float32, requires_grad=True).unsqueeze(0)

                # Defender's turn
                print(f"Episode {episode + 1}: Defender's turn.")
                defender_action_probs = defender_net(state_defender)
                masked_defender_action_probs, _ = mask_invalid_cards(defender_action_probs, game.players[1], deck)
                defender_card_index = torch.argmax(masked_defender_action_probs).item()
                chosen_defender_card = game.index_to_card(defender_card_index)
                # print(f"Defender's cards: {game.players[1]}")
                # print(f"Valid defense cards: {[card for card in game.players[1] if card in deck]}")
                # print(f"Chosen defender card: {chosen_defender_card}")

                if chosen_defender_card not in game.players[1]:
                    # Invalid move, defender loses
                    reward_attacker = 100
                    reward_defender = -100
                    done = True
                    print(f"Episode {episode + 1}: Invalid move by defender.")
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

        # Store game data for visualization
        game_data.append((attack_cards, defense_cards))
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

    for i, (attack_cards, defense_cards) in enumerate(game_data):
        ax = axes[i] if len(game_data) > 1 else axes
        ax.set_title(f"Game {i + 1}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1, max(len(attack_cards), len(defense_cards)))
        ax.set_ylim(-1, 2)

        # Plot attacking cards
        for j, card in enumerate(attack_cards):
            ax.text(j, 1, f"{card[0]} of {card[1]}", ha='center', va='center', fontsize=12, color='red')

        # Plot defending cards
        for j, card in enumerate(defense_cards):
            ax.text(j, 0, f"{card[0]} of {card[1]}", ha='center', va='center', fontsize=12, color='blue')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_networks()