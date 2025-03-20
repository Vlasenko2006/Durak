import torch
import torch.nn.functional as F
from durak_game import DurakGame
from neural_networks import attacker_net, defender_net, attacker_optimizer, defender_optimizer

# Hyperparameters
gamma = 0.99
batch_size = 64
num_episodes = 1000

def train_step():
    # Placeholder for training logic
    pass

def train_networks():
    for episode in range(num_episodes):
        game = DurakGame()
        state_attacker = torch.tensor(game.get_state(0), dtype=torch.float32).unsqueeze(0)
        state_defender = torch.tensor(game.get_state(1), dtype=torch.float32).unsqueeze(0)
        done = False

        while not done:
            # Attacker's turn
            state_attacker.requires_grad = True
            attacker_action = attacker_net(state_attacker)
            attacker_card_index = torch.argmax(attacker_action).item()

            if game.index_to_card(attacker_card_index) not in game.players[0]:
                # Invalid move, attacker loses
                reward_attacker = -100
                reward_defender = 100
                done = True
                defender_action = torch.zeros(36, requires_grad=True)  # Initialize defender_action to avoid unbound error
            else:
                # Valid move, proceed with defense
                game.players[0].remove(game.index_to_card(attacker_card_index))
                state_defender = torch.tensor(game.get_state(1), dtype=torch.float32).unsqueeze(0)
                state_defender.requires_grad = True

                # Defender's turn
                defender_action = defender_net(state_defender)
                defender_card_index = torch.argmax(defender_action).item()

                if game.index_to_card(defender_card_index) not in game.players[1]:
                    # Invalid move, defender loses
                    reward_attacker = 100
                    reward_defender = -100
                    done = True
                else:
                    # Valid defense, proceed with next round
                    game.players[1].remove(game.index_to_card(defender_card_index))
                    state_attacker = torch.tensor(game.get_state(0), dtype=torch.float32).unsqueeze(0)
                    state_attacker.requires_grad = True

                    # Check win conditions and rewards
                    if not game.players[0]:
                        reward_attacker = 100
                        reward_defender = -100
                        done = True
                    elif not game.players[1]:
                        reward_attacker = -100
                        reward_defender = 100
                        done = True

        # Ensure attacker_action and defender_action have consistent sizes
        target_attacker = torch.full_like(attacker_action, reward_attacker, dtype=torch.float32)
        target_defender = torch.full_like(defender_action, reward_defender, dtype=torch.float32)

        # Update networks with rewards
        attacker_optimizer.zero_grad()
        defender_optimizer.zero_grad()

        loss_attacker = F.mse_loss(attacker_action, target_attacker)
        loss_defender = F.mse_loss(defender_action, target_defender)

        loss_attacker.backward(retain_graph=True)
        loss_defender.backward()

        attacker_optimizer.step()
        defender_optimizer.step()

        print(f"Episode: {episode + 1}, Reward Attacker: {reward_attacker}, Reward Defender: {reward_defender}")

if __name__ == "__main__":
    train_networks()