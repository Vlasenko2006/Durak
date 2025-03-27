import torch
from durak_game import DurakGame
from visualize_games import visualize_games, moving_mean
from gameset import gameset
from neural_networks import CardNN
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim


#torch.autograd.set_detect_anomaly(True)

# Hyperparameters
gamma = 0.99
batch_size = 16
num_episodes = 2000
num_games_to_visualize = 3
reward_value = torch.tensor([1.], dtype=torch.float32, requires_grad=True)
margin_attacker = 0.
margin_defender = 0. 

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
accumulate_grad_att = np.zeros(num_episodes)
accumulate_grad_def = np.zeros(num_episodes)

attack_flag = torch.tensor([1.], dtype=torch.float32, requires_grad=True).unsqueeze(0)
defend_flag = -1 * torch.tensor([1.], dtype=torch.float32, requires_grad=True).unsqueeze(0)






lr = 1e-4
# Optimizers for both networks







def train_networks():
    player_0 = CardNN()   # attacker
    player_1 = CardNN()
    players = [player_0,player_1]
    player_0_optimizer = optim.Adam(player_0.parameters(), lr=lr)
    player_1_optimizer = optim.Adam(player_1.parameters(), lr=lr)
    margin_attacker = 0.
    margin_defender = 0. 
    
    global game_log  # Ensure we use the global variable
    # Define the deck of cards
    deck = [(rank, suit) for rank in ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] for suit in ['clubs', 'diamonds', 'hearts', 'spades']]
    
    for episode in range(num_episodes):
        
        print("Starting")
        
        if episode % 50 == 0 and episode > 0 and margin_attacker<0.5:
            print("margin_attacker = ", margin_attacker)
            margin_attacker += 0.05
            margin_defender += 0.05
            print("margin_attacker = ", margin_attacker)
            accumulate_grad_att
            att_to_show = accumulate_grad_att
            att_to_show[episode:] = np.nan
            plt.plot(moving_mean(att_to_show, window_size=5))
            plt.show()
            def_to_show = accumulate_grad_def
            def_to_show[episode:] = np.nan
            plt.plot(moving_mean(def_to_show, window_size=5))
            plt.show()
        
        # Zero gradients at the start of each batch
        player_0_optimizer.zero_grad()
        player_1_optimizer.zero_grad()
        accumulated_loss_defender = torch.tensor([0.], dtype=torch.float32, requires_grad=True)
        accumulated_loss_attacker = torch.tensor([0.], dtype=torch.float32, requires_grad=True)
        
        for batch in range(batch_size):
            game = DurakGame()
            if 'game_log' in locals(): 
                print("Exists")
                del game_log
        
            loss_attacker_loc, loss_defender_loc,\
            game_log, players = gameset(game,
                        players,
                        attack_flag,
                        defend_flag,
                        deck,
                        episode, 
                        game_log,
                        reward_value,
                        margin_attacker,
                        margin_defender,
                        gamma=0.99
                        )
                        
            accumulated_loss_attacker = accumulated_loss_attacker + loss_attacker_loc / batch_size
            accumulated_loss_defender = accumulated_loss_defender + loss_defender_loc / batch_size           
        
        print(f"num_episodes {episode} Accumulated Losses:", accumulated_loss_attacker, accumulated_loss_defender)

        # Perform backward pass after accumulating the losses
        accumulated_loss_attacker.backward(retain_graph=True)
        accumulated_loss_defender.backward(retain_graph=True)

        # Update networks with accumulated gradients
        player_0_optimizer.step()
        player_1_optimizer.step()
        
        accumulate_grad_att[episode] = accumulated_loss_attacker.item()
        accumulate_grad_def[episode] = accumulated_loss_defender.item()



        if episode % 100 == 0 and episode > 0:
            torch.save(players[0].state_dict(), "attacker_" + str(episode))
            torch.save(players[1].state_dict(), "attacker_" + str(episode))
            np.save("accumulate_grad_att_" + str(episode) , accumulate_grad_att)
            np.save("accumulate_grad_def_" + str(episode) , accumulate_grad_def)


        if episode % 20 == 0 and episode > 0:
            plt.plot(moving_mean(accumulate_grad_att[: episode], window_size=5))
            plt.title("Attacker " + str(episode))
            plt.show()
    
            plt.plot(moving_mean(accumulate_grad_def[: episode], window_size=5))
            plt.title("Defender " + str(episode))
            plt.show()
    
    return game_log, accumulate_grad_att, accumulate_grad_def

if __name__ == "__main__":
    game_log, accumulate_grad_att, accumulate_grad_def = train_networks()
    #visualize_games(game_log)