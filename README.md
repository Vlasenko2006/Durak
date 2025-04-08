# Durak Game with Reinforcement Learning
# Durak Game with Adversarial Reinforcement Learning

Welcome to the Durak Game with Reinforcement Learning! In this repository, you'll find an implementation of the traditional shedding-type Russian card game Durak (The Fool), enhanced with adversarial reinforcement learning for training neural networks to play the game. In short: two neural networks learn to play the Durak card game against each other. A neural network gets a reward if it moves by the game rules and a penalty in the opposite case. The goal is to get rid of all your cards, and the neural network that fits the goal gets an additional reward. The game has an element of randomness as the cards are scrambled in the deck before it starts. Because of this randomness, the neural networks must elaborate their strategies: which cards to keep at the beginning of the game (strong cards, or trumps) and which ones to give up immediately. 

This repository has a Gameplay.py GUI, allowing you to live play against one of the neural networks.  Below is an example of how this GUI looks like:


![Example of a fameplay](https://github.com/Vlasenko2006/Durak/blob/main/Durak_game.png)

## Gameplay Rules

Here are the basic rules:

1. **Deck and Players**: The game uses a deck of 36 cards (6 to Ace in each suit). It can be played with 2 or more players.
2. **Trump Card**: The last card dealt is placed face up and its suit becomes the trump suit for the game. The remaining deck is placed on top of it.
3. **Attacking and Defending**: Players take turns attacking and defending. The attacker plays a card, and the defender must play a higher card of the same suit or a trump card to beat it.
4. **Shedding Cards**: The attacker can continue to play cards of the same rank as any already played card. The defender must beat each card.
5. **End of Turn**: If the defender successfully beats all cards, the turn ends and the next player becomes the attacker. If the defender cannot beat a card, they must take all the cards on the table, and the turn ends.
6. **Winning and Losing**: The game continues until the deck is exhausted and one player remains with cards. This player is the "durak".


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

- **training_batch**  is a top level script. It initializes the game, sets up the neural networks and optimizers, and runs a training loop for a specified number of episodes. It includes functions for saving and loading model checkpoints, managing the game state, and updating the neural networks based on the rewards received from the game outcomes.

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

#### Neural Network

The neural network is defined in the `CardNN` class, which is a PyTorch neural network model. All but one its inputs correspond to the card's index in the deck. Each input gets one if thge corresponding card is in the player's hand or lies on the table and zero in the opposite case. The last input is the attacker flag. It equals one if the NN attacks or -1 if it defends. The NN's outputs are card's probabilites $p$. The card having higest probability is chosen for an attack/defence.   

### Reinforcement Learning

The reinforcement learning aspect is implemented using neural networks that make decisions for the attacker and defender agents. These networks are trained using the reward signals derived from the game outcomes.


### Rewards


- **reward**: Computes the rewards for the attacker and defender based on the game outcomes, updates the game log, and returns the updated state. It consists of two parts: constant rewards for attacker/defender $R_A$ and $R_D$ and their updates depending on how certain are the neural networks about their decisions.
- Update for attacker 

$$
R_A = R_A - \log(p_{attacker}) + \log \left( 1 + mean(p_{rest}) \right)
$$

- Update for defender

$$
R_D = R_D - \log (p_{defender}) + \log\left(1 + mean(p_{rest})\right)
$$

here $p_{attacker}$ and $p_{defender}$ are the probabilities of the attacking and defending cards respectively. The attack/defence probabilities for  the remaining cards in hands are and $p_{rest}$ 


- **gameset**: Computes the total rewards for the entire game applying Bellman's equation, passing the resut to the MSE cost function. This script refills hands according to the game rules.  


### Training's Visualization

Training visualization is done in the top level script **training_batch**. It plots cost values for the attacker and defender NNs, and the probabilities of attacking and defending cards. To smootheng the training curves I used the 
- **moving_mean**: Computes the moving mean of a given array, used for smoothing visualization curves.


Below are the convergence curves of the training process, showing the accumulated gradients for the attacker

![Attacker Convergence](https://github.com/Vlasenko2006/Durak/blob/main/attacker_13800.png)

and defender

![Defender Convergence](https://github.com/Vlasenko2006/Durak/blob/main/defender_13800.png))

Note the behavior of attackers'/ defender's cost function. The defender's sharp drop in the first 1000 games is due to training the confidence in the chosen card rather than learning the game rules. Recall the confidence logarithmic terms in the rewards: when the neural network is unsure of its choice, these logarithms give the main contribution to the value of the cost function. When the neural network becomes confident in its choice (somewhere after 1000 games), its value drops and the neural network starts learning to play and win. Note that the value of the cost function of the defending neural network grows as the attacking network learns. 



## Example Usage

To train the neural networks, run the `training_batch.py` script:

```sh
python training_batch.py
```



