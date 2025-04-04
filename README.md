# Durak Game with Reinforcement Learning

This repository contains an implementation of the card game Durak, enhanced with reinforcement learning for training neural networks to play the game. The code uses PyTorch for neural network modeling and training.

## Table of Contents
- [Installation](#installation)
- [Training the Networks](#training-the-networks)
- [Game Mechanics](#game-mechanics)
- [Visualizing Training Results](#visualizing-training-results)

## Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/Vlasenko2006/Durak.git
   cd Durak
   ```

2. **Create a Conda environment**:
   ```sh
   conda create --name durak_env python=3.9
   conda activate durak_env
   ```

3. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

4. **Install additional dependencies** (if necessary):
   ```sh
   conda install -c conda-forge matplotlib
   conda install -c conda-forge pillow
   ```

## Training the Networks

To train the neural networks, run the `training_batch.py` script. This script trains the neural networks for both the attacker and defender agents using reinforcement learning.

### Training Script

This script initializes the game, sets up the neural networks and optimizers, and runs a training loop for a specified number of episodes. It includes functions for saving and loading model checkpoints, managing the game state, and updating the neural networks based on the rewards received from the game outcomes.

- **save_checkpoint**: Saves the current state of the model, optimizer, and training progress to a file.
- **load_checkpoint**: Loads the state of the model, optimizer, and training progress from a file.
- **find_last_checkpoint**: Finds the most recent checkpoint file in the outputs directory.
- **train_networks**: Main function to train the neural networks. It initializes the game, sets up the neural networks and optimizers, and runs the training loop.

## Game Mechanics

The game logic is implemented in the `DurakGame` class, which handles the deck creation, dealing cards, drawing cards, and managing the state of the game.

### Durak Game Class

This class defines the main mechanics of the Durak game, including deck creation, dealing cards, drawing cards, and determining the game state.

- **__init__**: Initializes the game by creating the deck, dealing cards to players, and setting the trump card.
- **get_deck**: Returns a list of all possible cards in the deck.
- **create_deck**: Shuffles the deck and returns it.
- **deal_cards**: Deals 6 cards to each player from the deck.
- **draw_card**: Draws a card from the deck for the specified player.
- **refill_hands**: Refills the players' hands to have 6 cards each, if possible.
- **get_state**: Returns the current state of a player's hand as a list of binary values.
- **update_state**: Updates the game state based on the defender's action.
- **card_to_index**: Converts a card to its corresponding index in the state representation.
- **index_to_card**: Converts an index in the state representation to its corresponding card.
- **can_beat**: Determines if a defending card can beat an attacking card.
- **indexes_to_cards**: Converts a list of card indexes to their corresponding cards.
- **get_player_with_smallest_card**: Determines which player has the smallest card and sets the attack flag accordingly.

### Reinforcement Learning

The reinforcement learning aspect is implemented using neural networks that make decisions for the attacker and defender agents. These networks are trained using the reward signals derived from the game outcomes.

#### Neural Network

The neural network is defined in the `CardNN` class, which is a PyTorch neural network model.

- **__init__**: Initializes the neural network layers and normalization layers.
- **forward**: Defines the forward pass of the neural network, including the application of the softmax function for output probabilities.

### Rewards

The reward function assigns positive or negative rewards based on the game outcomes, guiding the learning process of the neural networks.

- **rewards**: Computes the rewards for the attacker and defender based on the game outcomes, updates the game log, and returns the updated state.

### Visualizing Training Results

The script includes functions for visualizing the training results, such as the accumulated gradients and the average card probabilities.

- **visualize_games**: Visualizes the game log and training progress.
- **moving_mean**: Computes the moving mean of a given array, used for smoothing visualization curves.

## Example Usage

To train the neural networks, run the `training_batch.py` script:

```sh
python training_batch.py
```

To visualize the training results, you can use the `visualize_games` function on the saved game log.

## Convergence Curves

Below are the convergence curves of the training process, showing the accumulated gradients for the attacker and defender:

![Attacker Convergence](images/attacker_convergence.png)
![Defender Convergence](images/defender_convergence.png)

The curves show the learning progress of the neural networks over time, with a moving mean applied for smoothing.
