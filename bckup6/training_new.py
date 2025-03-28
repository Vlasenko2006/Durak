import torch
import torch.nn.functional as F
from durak_game import DurakGame
from neural_networks import attacker_net, defender_net, attacker_optimizer, defender_optimizer
import matplotlib.pyplot as plt
from game_turns import game_turns
from visualize_games import visualize_games

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
game_log = []  # Variable to hold all game steps



def train_networks():
    global game_log  # Ensure we use the global variable
    # Define the deck of cards
    deck = [(rank, suit) for rank in ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] for suit in ['clubs', 'diamonds', 'hearts', 'spades']]
    
    game = None  # Initialize game to None

    for episode in range(num_episodes):
        game = DurakGame()
        state_attacker = torch.tensor(game.get_state(0), dtype=torch.float32, requires_grad=True).unsqueeze(0)
        state_defender = torch.tensor(game.get_state(1), dtype=torch.float32, requires_grad=True).unsqueeze(0)

        step_number, reward_attacker, reward_defender, attack_cards, defense_cards, game_log, done = game_turns(
            game, state_attacker, state_defender, deck, episode, game_log, attacker_net, defender_net
        )

        # Store game data for visualization
        game_data.append((attack_cards, defense_cards, game.trump_suit, step_number, reward_attacker, reward_defender, list(game.players[0]), list(game.players[1])))
        if len(game_data) > num_games_to_visualize:
            game_data.pop(0)

        # Ensure attacker_action and defender_action have consistent sizes
        target_attacker = torch.full_like(attacker_net(state_attacker), reward_attacker, dtype=torch.float32)
        target_defender = torch.full_like(defender_net(state_defender), reward_defender, dtype=torch.float32)

        # Update networks with rewards
        attacker_optimizer.zero_grad()
        defender_optimizer.zero_grad()

        loss_attacker = F.mse_loss(attacker_net(state_attacker), target_attacker)
        loss_defender = F.mse_loss(defender_net(state_defender), target_defender)

        loss_attacker.backward(retain_graph=True)
        loss_defender.backward()

        attacker_optimizer.step()
        defender_optimizer.step()

        print(f"Episode: {episode + 1}, Reward Attacker: {reward_attacker}, Reward Defender: {reward_defender}")

    # Visualize the last 3 games
    #print("game_data =", game_data)
    visualize_games(game_log)

    return game_log

if __name__ == "__main__":
    game_log = train_networks()
   # for log in game_log:
  #      print(log)
