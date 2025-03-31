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
import glob
import os

# Hyperparameters
gamma = 0.5
batch_size = 4
num_episodes = 20000
num_games_to_visualize = 3
reward_value = torch.tensor([1.], dtype=torch.float32, requires_grad=True)

# Store game data for visualization
game_data = []
game_log = []  # Variable to hold all game steps
accumulate_grad_att = np.zeros(num_episodes)
accumulate_grad_def = np.zeros(num_episodes)
avg_attacker_card_probs = np.zeros(num_episodes)
avg_defender_card_probs = np.zeros(num_episodes)
avg_mean_masked_attacker_probs = np.zeros(num_episodes)
avg_mean_masked_defender_probs = np.zeros(num_episodes)

attack_flag = torch.tensor([1.], dtype=torch.float32, requires_grad=True).unsqueeze(0)
defend_flag = -1 * torch.tensor([1.], dtype=torch.float32, requires_grad=True).unsqueeze(0)

lr = 2 * 1e-4

# Function to save model, optimizer state, and arrays
def save_checkpoint(state, filename):
    torch.save(state, filename)

# Function to load model, optimizer state, and arrays
def load_checkpoint(filename, player_0, player_1, player_0_optimizer, player_1_optimizer):
    checkpoint = torch.load(filename)
    player_0.load_state_dict(checkpoint['player_0_state_dict'])
    player_1.load_state_dict(checkpoint['player_1_state_dict'])
    player_0_optimizer.load_state_dict(checkpoint['player_0_optimizer'])
    player_1_optimizer.load_state_dict(checkpoint['player_1_optimizer'])
    return checkpoint['episode']

# Function to find the last saved checkpoint
def find_last_checkpoint():
    checkpoints = glob.glob("outputs/checkpoint_*.pth.tar")
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
    return latest_checkpoint

# Optimizers for both networks
def train_networks(load_model=False):
    player_0 = CardNN()   # attacker
    player_1 = CardNN()
    players = [player_0, player_1]
    player_0_optimizer = optim.Adam(player_0.parameters(), lr=lr)
    player_1_optimizer = optim.Adam(player_1.parameters(), lr=lr)
    margin_attacker = 0.5
    margin_defender = 0.5 
    
    global game_log, accumulate_grad_att, accumulate_grad_def, avg_attacker_card_probs, avg_defender_card_probs, avg_mean_masked_attacker_probs, avg_mean_masked_defender_probs  # Ensure we use the global variables
    # Define the deck of cards
    deck = [(rank, suit) for rank in ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] for suit in ['clubs', 'diamonds', 'hearts', 'spades']]
    
    start_episode = 0

    if load_model:
        last_checkpoint = find_last_checkpoint()
        if last_checkpoint:
            start_episode = load_checkpoint(last_checkpoint, player_0, player_1, player_0_optimizer, player_1_optimizer)
            episode_num = int(last_checkpoint.split('_')[1].split('.')[0])
            accumulate_grad_att = np.load(f"outputs/accumulate_grad_att_{episode_num}.npy")
            accumulate_grad_def = np.load(f"outputs/accumulate_grad_def_{episode_num}.npy")
            avg_attacker_card_probs = np.load(f"outputs/avg_attacker_card_probs_{episode_num}.npy")
            avg_defender_card_probs = np.load(f"outputs/avg_defender_card_probs_{episode_num}.npy")
            avg_mean_masked_attacker_probs = np.load(f"outputs/avg_mean_masked_attacker_probs_{episode_num}.npy")
            avg_mean_masked_defender_probs = np.load(f"outputs/avg_mean_masked_defender_probs_{episode_num}.npy")

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

        # Extract attacker and defender card probabilities from game_log
        attacker_card_probs_episode = [log['attacker_card_prob'] for log in game_log if 'attacker_card_prob' in log]
        defender_card_probs_episode = [log['defender_card_prob'] for log in game_log if 'defender_card_prob' in log]
        mean_masked_attacker_action_probs_episode = [log['mean_masked_attacker_action_probs'] for log in game_log if 'mean_masked_attacker_action_probs' in log]
        mean_masked_defender_action_probs_episode = [log['mean_masked_defender_action_probs'] for log in game_log if 'mean_masked_defender_action_probs' in log]

        # Compute means and save them into numpy arrays
        if attacker_card_probs_episode:
            avg_attacker_card_probs[episode] = np.mean(attacker_card_probs_episode)
        if defender_card_probs_episode:
            avg_defender_card_probs[episode] = np.mean(defender_card_probs_episode)
        if mean_masked_attacker_action_probs_episode:
            avg_mean_masked_attacker_probs[episode] = np.mean(mean_masked_attacker_action_probs_episode)
        if mean_masked_defender_action_probs_episode:
            avg_mean_masked_defender_probs[episode] = np.mean(mean_masked_defender_action_probs_episode)

        if episode % 20 == 0 and episode > 0:
            
            save_checkpoint({
                'episode': episode,
                'player_0_state_dict': player_0.state_dict(),
                'player_1_state_dict': player_1.state_dict(),
                'player_0_optimizer': player_0_optimizer.state_dict(),
                'player_1_optimizer': player_1_optimizer.state_dict(),
            }, filename=f"outputs/checkpoint_{episode}.pth.tar")
            np.save(f"outputs/accumulate_grad_att_{episode}", accumulate_grad_att)
            np.save(f"outputs/accumulate_grad_def_{episode}", accumulate_grad_def)


            np.save(f"outputs/avg_attacker_card_probs_{episode}", avg_attacker_card_probs)
            np.save(f"outputs/avg_defender_card_probs_{episode}", avg_defender_card_probs)
            np.save(f"outputs/avg_mean_masked_attacker_probs_{episode}", avg_mean_masked_attacker_probs)
            np.save(f"outputs/avg_mean_masked_defender_probs_{episode}", avg_mean_masked_defender_probs)
            
            plt.plot(moving_mean(accumulate_grad_att[:episode], window_size=15))
            plt.title("Attacker " + str(episode))
            plt.show()
    
            plt.plot(moving_mean(accumulate_grad_def[:episode], window_size=15))
            plt.title("Defender " + str(episode))
            plt.show()

            # Plot averaged values
            plt.plot(moving_mean(avg_attacker_card_probs[:episode], window_size=15))
            plt.title("Averaged Attacker Card Probabilities")
            plt.show()
            
            plt.plot(moving_mean(avg_defender_card_probs[:episode], window_size=15))
            plt.title("Averaged Defender Card Probabilities")
            plt.show()

            plt.plot(moving_mean(avg_mean_masked_attacker_probs[:episode], window_size=15))
            plt.title("Averaged Mean Masked Attacker Action Probabilities")
            plt.show()
            
            plt.plot(moving_mean(avg_mean_masked_defender_probs[:episode], window_size=15))
            plt.title("Averaged Mean Masked Defender Action Probabilities")
            plt.show()
    
    return game_log, accumulate_grad_att, accumulate_grad_def

if __name__ == "__main__":
    load_model = True # Set to True if you want to load the model and optimizer state
    game_log, accumulate_grad_att, accumulate_grad_def = train_networks(load_model=load_model)
    #visualize_games(game_log)