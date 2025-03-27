import torch
from durak_game import DurakGame
from visualize_games import visualize_games, moving_mean
from gameset import gameset
from neural_networks import CardNN
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import time  # Import the time module
from compute_one_batch import compute_one_batch

#torch.autograd.set_detect_anomaly(True)

# Hyperparameters
gamma = 0.99
batch_size = 16
num_episodes = 2000
num_games_to_visualize = 3
reward_value = torch.tensor([1.], dtype=torch.float32, requires_grad=True)
margin_attacker = 0.
margin_defender = 0. 

# Store game data for visualization
game_data = []
game_log = []  # Variable to hold all game steps
accumulate_grad_att = np.zeros(num_episodes)
accumulate_grad_def = np.zeros(num_episodes)

attack_flag = torch.tensor([1.], dtype=torch.float32, requires_grad=True).unsqueeze(0)
defend_flag = -1 * torch.tensor([1.], dtype=torch.float32, requires_grad=True).unsqueeze(0)

lr = 1e-4

# Function to save model and optimizer state
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

# Function to load model and optimizer state
def load_checkpoint(filename, player_0, player_1, player_0_optimizer, player_1_optimizer):
    checkpoint = torch.load(filename)
    player_0.load_state_dict(checkpoint['player_0_state_dict'])
    player_1.load_state_dict(checkpoint['player_1_state_dict'])
    player_0_optimizer.load_state_dict(checkpoint['player_0_optimizer'])
    player_1_optimizer.load_state_dict(checkpoint['player_1_optimizer'])
    return checkpoint['episode']

# Optimizers for both networks
def train_networks(load_model=False):
    player_0 = CardNN()   # attacker
    player_1 = CardNN()
    players = [player_0, player_1]
    player_0_optimizer = optim.Adam(player_0.parameters(), lr=lr)
    player_1_optimizer = optim.Adam(player_1.parameters(), lr=lr)
    margin_attacker = 0.
    margin_defender = 0. 
    
    global game_log  # Ensure we use the global variable
    # Define the deck of cards
    deck = [(rank, suit) for rank in ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] for suit in ['clubs', 'diamonds', 'hearts', 'spades']]
    
    start_episode = 0

    if load_model:
        start_episode = load_checkpoint("checkpoint.pth.tar", player_0, player_1, player_0_optimizer, player_1_optimizer)
    
    for episode in range(start_episode, num_episodes):
        
        print("Starting")
        
        if episode % 5 == 0 and episode > 0 and margin_attacker < 0.48:
            print("margin_attacker = ", margin_attacker)
            margin_attacker += 0.05
            margin_defender += 0.05
            print("margin_attacker = ", margin_attacker)
        
        
        start_time = time.time()  # Start timing the batch computation  
        
        game_log, players, accumulated_loss_attacker, \
         accumulated_loss_defender, \
         player_0_optimizer, player_1_optimizer  = compute_one_batch(player_0_optimizer,
                              player_1_optimizer,
                              batch_size,
                              players,
                              attack_flag,
                              defend_flag,
                              episode,
                              deck,
                              game_log,
                              reward_value,
                              margin_attacker,
                              margin_defender,
                              )
        
        end_time = time.time()  # End timing the batch computation
        batch_time = end_time - start_time  # Calculate the time taken for the batch
        print(f"Time taken for batch {batch_size}: {batch_time:.4f} seconds")
        
        accumulate_grad_att[episode] = accumulated_loss_attacker.item()
        accumulate_grad_def[episode] = accumulated_loss_defender.item()

        if episode % 100 == 0 and episode > 0:
            save_checkpoint({
                'episode': episode,
                'player_0_state_dict': player_0.state_dict(),
                'player_1_state_dict': player_1.state_dict(),
                'player_0_optimizer': player_0_optimizer.state_dict(),
                'player_1_optimizer': player_1_optimizer.state_dict(),
            }, filename="checkpoint.pth.tar")
            np.save("accumulate_grad_att_" + str(episode), accumulate_grad_att)
            np.save("accumulate_grad_def_" + str(episode), accumulate_grad_def)

        if episode % 20 == 0 and episode > 0:
            plt.plot(moving_mean(accumulate_grad_att[:episode], window_size=5))
            plt.title("Attacker " + str(episode))
            plt.show()
    
            plt.plot(moving_mean(accumulate_grad_def[:episode], window_size=5))
            plt.title("Defender " + str(episode))
            plt.show()
    
    return game_log, accumulate_grad_att, accumulate_grad_def

if __name__ == "__main__":
    load_model = False  # Set to True if you want to load the model and optimizer state
    game_log, accumulate_grad_att, accumulate_grad_def = train_networks(load_model=load_model)
    #visualize_games(game_log)