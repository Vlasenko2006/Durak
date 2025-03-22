import torch
import torch.nn.functional as F
from durak_game import DurakGame
from neural_networks import attacker_net, defender_net, attacker_optimizer, defender_optimizer
import matplotlib.pyplot as plt
from game_turns import game_turns
from visualize_games import visualize_games
import numpy as np

# Hyperparameters
gamma = 0.99
batch_size = 64
num_episodes = 3
num_games_to_visualize = 3
reward_value = torch.tensor([1.], dtype=torch.float32, requires_grad=True)

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

attack_flag = torch.tensor([1.], dtype=torch.float32, requires_grad=True).unsqueeze(0)
defend_flag = -1 * torch.tensor([1.], dtype=torch.float32, requires_grad=True).unsqueeze(0)


def train_networks():
    global game_log  # Ensure we use the global variable
    # Define the deck of cards
    deck = [(rank, suit) for rank in ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] for suit in ['clubs', 'diamonds', 'hearts', 'spades']]
    
    game = None  # Initialize game to None

    for episode in range(num_episodes):
        game = DurakGame()
        
        # attacker_ID
        state_attacker = torch.tensor(game.get_state(0), dtype=torch.float32, requires_grad=True).unsqueeze(0)
        state_defender = torch.tensor(game.get_state(1) , dtype=torch.float32, requires_grad=True).unsqueeze(0)
        
        
        print(state_attacker.shape)
        


        attacker = 0
        defender = 1
        print('\n')
        print('=================')
        
        played_cards = torch.zeros(1,36)
        taken_cards = torch.zeros(2,36)
        Q_attacker_previous = Q_defender_previous = torch.tensor([1.], dtype=torch.float32, requires_grad=True)

####################  One turn game
##################### Need a loop over this turn, :: while deck_status >0 and (game.players[attacker] and game.players[attacker])

        big_loop_done = False
        counter = 1
        while not big_loop_done:
            
            played_cards, reward_attacker, reward_defender, output_defender, output_attacker, game_log, done = game_turns( game, 
                                                                                                          attacker,
                                                                                                          defender,
                                                                                                          attack_flag,
                                                                                                          defend_flag,
                                                                                                          played_cards,
                                                                                                          taken_cards,
                                                                                                          state_attacker,
                                                                                                          state_defender,
                                                                                                          deck,
                                                                                                          episode, 
                                                                                                          game_log,
                                                                                                          attacker_net,
                                                                                                          defender_net,
                                                                                                          reward_value,
                                                                                                          game_data = game_data
                                                                                                          )
            
            deck_status = game.refill_hands(attacker,defender)
            if deck_status == 0:
                if not game.players[attacker]:
                    reward_attacker = reward_attacker + 3 * reward_value
                    big_loop_done = True
                if not game.players[attacker]:
                    reward_defender = reward_defender + 3 * reward_value
                    big_loop_done = True
                    
            if any(log_entry['result'] == "Wrong card chosen" for log_entry in game_log): big_loop_done = True
            
            if np.mod(counter,2) == 0:
                attacker = 1
                defender = 0
            else:
                attacker = 0
                defender = 1
            
            
            
            if reward_attacker != 0: reward_attacker = reward_attacker/reward_attacker
            if reward_defender != 0: reward_defender = reward_defender/reward_defender
            
            
            print("reward_attacker.dtype = ", reward_attacker.dtype)
            print("reward_defender.dtype = ", reward_defender.dtype)

            # Calculate target Q-values using Bellman's equation
            if counter == 1:
                target_attacker = gamma * reward_attacker
                target_defender = gamma * reward_defender            
            else:
                target_attacker = Q_attacker_previous + gamma * reward_attacker
                target_defender = Q_defender_previous + gamma * reward_defender
            
            print("target_attacker.dtype = ", target_attacker.dtype)
            print("target_defender.dtype = ", target_defender.dtype)
            print("Q_attacker_previous.dtype = ", Q_attacker_previous.dtype)
            print("Q_defender_previous.dtype = ", Q_defender_previous.dtype)
            
            # Calculate loss
            loss_attacker = F.mse_loss(target_attacker, Q_attacker_previous)
            loss_defender = F.mse_loss(target_defender, Q_defender_previous)
            
            # Update networks with rewards
            attacker_optimizer.zero_grad()
            defender_optimizer.zero_grad()

            loss_attacker.backward(retain_graph=True)
            loss_defender.backward()

            attacker_optimizer.step()
            defender_optimizer.step()

            print(f"Episode: {episode + 1}, Reward Attacker: {reward_attacker}, Reward Defender: {reward_defender}")
            
            # Update states
            Q_attacker_previous = reward_attacker
            Q_attacker_previous = reward_defender

        print(f"Episode: {episode + 1}, Reward Attacker: {reward_attacker}, Reward Defender: {reward_defender}")

    # Visualize the last 3 games
   # print("game_data =", game_data)
    visualize_games(game_log)

    return game_data #game_log

if __name__ == "__main__":
    game_gata= train_networks()
   # for log in game_log:
  #      print(log)
