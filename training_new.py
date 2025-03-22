import torch
from durak_game import DurakGame
from visualize_games import visualize_games
from gameset import gameset
from neural_networks import attacker_optimizer, defender_optimizer

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
    
    for episode in range(num_episodes):
        
        game = DurakGame()
        
        # Zero gradients at the start of each batch
        attacker_optimizer.zero_grad()
        defender_optimizer.zero_grad()
        accumulated_loss_defender = torch.tensor([0.], dtype=torch.float32, requires_grad=True)
        accumulated_loss_attacker = torch.tensor([0.], dtype=torch.float32, requires_grad=True)
        
        for batch in range(batch_size):
            
            print("batch_iter = ", batch)
        
            loss_attacker_loc, loss_defender_loc, game_log = gameset(game,
                        attack_flag,
                        defend_flag,
                        deck,
                        episode, 
                        game_log,
                        reward_value,
                        gamma=0.99
                        )
                        
            accumulated_loss_attacker = accumulated_loss_attacker + loss_attacker_loc / batch_size
            accumulated_loss_defender = accumulated_loss_defender + loss_defender_loc / batch_size           
        
        print("Accumulated Losses:", accumulated_loss_attacker.item(), accumulated_loss_defender.item())

        # Perform backward pass after accumulating the losses
        accumulated_loss_attacker.backward(retain_graph=True)
        accumulated_loss_defender.backward(retain_graph=True)

        # Update networks with accumulated gradients
        attacker_optimizer.step()
        defender_optimizer.step()


    # Visualize the last 3 games
    visualize_games(game_log)

    return game_data

if __name__ == "__main__":
    game_data = train_networks()