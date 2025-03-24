import torch

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


def defender_can_beat(defenders_cards, 
                      attack_card,
                      chosen_defender_card,
                      game, 
                      decision_to_defend,
                      margin_defender,
                      verbose=False):
    
  #  print("decision_to_defend , margin_defender", decision_to_defend , margin_defender)
    if decision_to_defend > margin_defender:
        defence_decision = "decide_to_defend"
        for defend_card in defenders_cards:
            if game.can_beat(attack_card, defend_card) and not game.can_beat(attack_card, chosen_defender_card):
                if verbose: print('Defender can beat, but chosen wrong card')
                defence_decision = "failure"
                break
    else:
        defence_decision = "withdraw"
        
   # print("ATTACK = ", attack_card, "DEFEND = ", chosen_defender_card, "Defenders cards", defenders_cards, "TRUMP ", game.trump_suit  )
    return defence_decision


def game_turns(game, 
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
               margin_attacker = 0.5,
               margin_defender = 0.5,
               verbose=False
               ):

    attack_value = None
    reward_attacker = torch.tensor([0.], dtype=torch.float32, requires_grad=True)
    reward_defender = torch.tensor([0.], dtype=torch.float32, requires_grad=True)
    done = False

    cards_on_a_table = torch.zeros(1, 36, dtype=torch.float32)  # Convert to Float
    not_playing_cards = played_cards.clone().float()  # Convert to Float to avoid type mismatch

    defender_action_probs = None  # Initialize defender_action_probs
    step_number = 0  # Initialize step number

    while not done:
        step_number += 1  # Increment step number

        # Attacker's turn
        valid_attacker_cards = [card for card in game.players[attacker] if card in deck]
        if attack_value:
            valid_attacker_cards = [card for card in valid_attacker_cards if card[0] == attack_value]

        if not valid_attacker_cards:
            done = True
            if verbose: print(f"Episode {episode + 1}: No valid cards to attack.")
            break

        # Debugging statement to check values
        # print(f"Before updating not_playing_cards: {not_playing_cards}")
        # print(f"Cards on table before attacker: {cards_on_a_table}")
        # print(f"Attacker's hand before play: {game.players[attacker]}")

        empty_index = torch.tensor([[-1]], dtype=torch.float32)

        output_attacker = attacker_net(state_attacker, attack_flag, played_cards, cards_on_a_table, empty_index)
        attacker_action_probs = output_attacker[..., :-1]
        decision_to_continue_attack = output_attacker[..., -1]

        masked_attacker_action_probs, mask = mask_invalid_cards(attacker_action_probs, valid_attacker_cards, deck)
        attacker_card_index = torch.argmax(masked_attacker_action_probs).item()
        chosen_card = game.index_to_card(attacker_card_index)\
            
        # print("attacker_card_index = ", attacker_card_index, "masked_attacker_action_probs", masked_attacker_action_probs)
        attacker_card_prob = masked_attacker_action_probs[0,attacker_card_index]

        if chosen_card not in game.players[attacker]:
            # Invalid move, attacker loses
            reward_attacker =  - torch.log(attacker_card_prob - 1)
            reward_defender = 0
            done = True
            defender_action_probs = torch.zeros_like(attacker_action_probs, requires_grad=True)
            if verbose: print(f"Episode {episode + 1}: Invalid move by attacker.")
            return played_cards, reward_attacker, reward_defender, output_defender, output_attacker, game_log, done
        else:
            # Valid move, proceed with defense
            game.players[attacker].remove(chosen_card)
            state_defender = torch.tensor(game.get_state(1), dtype=torch.float32, requires_grad=True).unsqueeze(0)
            cards_on_a_table = game.update_state(attacker, attacker_card_index, cards_on_a_table)

            if attack_value is None:
                attack_value = chosen_card[0]

            # print(f"Attacker played card: {chosen_card}")
            # print(f"Attacker's hand after play: {game.players[attacker]}")
            # print(f"Cards on table after attacker: {cards_on_a_table}")

            # Defender's turn
            if verbose: print(f"Episode {episode + 1}: Defender's turn.")
            attakers_card = torch.tensor([[attacker_card_index]], dtype=torch.float32)
            output_defender = defender_net(state_defender, defend_flag, played_cards, cards_on_a_table, attakers_card)
            defender_action_probs = output_defender[..., :-1]
            decision_to_defend = output_defender[..., -1]
            masked_defender_action_probs, _ = mask_invalid_cards(defender_action_probs, game.players[defender], deck)
            defender_card_index = torch.argmax(masked_defender_action_probs).item()
            chosen_defender_card = game.index_to_card(defender_card_index)
            
            
            defender_card_prob = masked_defender_action_probs[0,defender_card_index]
            
            
            


            cards_on_a_table = game.update_state(defender, defender_card_index, cards_on_a_table)

            defence_decision = defender_can_beat(game.players[defender],
                                                 chosen_card,
                                                 chosen_defender_card,
                                                 game,
                                                 decision_to_defend,
                                                 margin_defender
                                                 )
            
            if defence_decision == "failure":
                if verbose: print("Defence Failure")
                print("Defence Failure")
                reward_defender =  -reward_value * (1 + defender_card_prob) #torch.log(defender_card_prob - 1)
                game_log.append({
                    'episode': episode + 1,
                    'step': step_number,
                    'trump': game.trump,
                    'attacker_action': chosen_card,
                    'defender_action': chosen_defender_card,
                    'remaining_attacker_hand': list(game.players[attacker]),
                    'remaining_defender_hand': list(game.players[defender]),
                    'result': "Wrong card chosen"
                })
                done = True
                
                break
            elif defence_decision == "withdraw":
                print("Withdraw")
                print("RW = ", reward_value)
                reward_attacker = reward_value  * attacker_card_prob
                reward_defender = 0 * reward_defender
                game_log.append({
                    'episode': episode + 1,
                    'step': step_number,
                    'trump': game.trump,
                    'attacker_action': chosen_card,
                    'defender_action': chosen_defender_card,
                    'remaining_attacker_hand': list(game.players[attacker]),
                    'remaining_defender_hand': list(game.players[defender]),
                    'result': "Attacker wins 1"
                })
                done = True

                break

            if chosen_defender_card not in game.players[defender] or not game.can_beat(chosen_card, chosen_defender_card):
                reward_attacker = reward_value  * attacker_card_prob #reward_attacker + reward_value  # Do I need this?
                done = True
                winner = "Attacker wins 2"
                chosen_defender_card = None
                if verbose: print("Attacker wins from the first turn")
            else:
                game.players[defender].remove(chosen_defender_card)
               # game.players[defender].remove(chosen_defender_card)
                state_attacker = torch.tensor(game.get_state(0), dtype=torch.float32, requires_grad=True).unsqueeze(0)

                # print(f"Defender played card: {chosen_defender_card}")
                # print(f"Defender's hand after play: {game.players[defender]}")
                # print(f"Cards on table after defender: {cards_on_a_table}")

                # No more cards in hand remaining, defender / attacker wins
                if not game.players[attacker]:
                    reward_attacker = reward_value * torch.log(attacker_card_prob + 1)
                    done = True
                    winner = "Attacker wins, no cards remaining"
                    played_cards = not_playing_cards.clone()
                elif not game.players[defender]:
                    reward_defender = reward_value * torch.log(defender_card_prob + 1)
                    done = True
                    winner = "Defender wins, no cards remaining"
                    played_cards = not_playing_cards.clone()
                elif not game.players[defender] and not game.players[attacker]:
                    reward_attacker =  reward_value  * torch.log(attacker_card_prob + 1)
                    reward_defender =  reward_value  * torch.log(defender_card_prob + 1)
                    done = True
                    winner = "No winner, no cards remaining"
                    played_cards = not_playing_cards.clone()
                else:
                    reward_defender = reward_value * torch.log(defender_card_prob + 1)
                    winner = "Defender wins 2"
            

            print(winner)
            
            if verbose: print(f"Episode {episode + 1}:" + winner)
            game_log.append({
                'episode': episode + 1,
                'step': step_number,
                'trump': game.trump,
                'attacker_action': chosen_card,
                'defender_action': chosen_defender_card,
                'remaining_attacker_hand': list(game.players[attacker]),
                'remaining_defender_hand': list(game.players[defender]),
                'result': winner
            })

            if not any(card[0] == attack_value for card in game.players[attacker]):
                attack_value = None
                done = True
                if verbose: print(f"Episode {episode + 1}: Attacker has no more cards of the same value.")
                break

            # Attacker decides whether to continue attack
            continue_attack = decision_to_continue_attack > margin_attacker  # Randomly decide to continue or stop
            if not continue_attack:
                played_cards = not_playing_cards.clone()
                done = True
                if verbose: print(f"Episode {episode + 1}: Attacker decides to stop the attack.")
                break
            # if winner == "Defender wins": reward_defender 
            # if winner == "Attacker wins": reward_attacker


        # Update not_playing_cards to include cards on the table without duplication
        not_playing_cards = torch.logical_or(not_playing_cards, cards_on_a_table)
    print("RFEWARD DEFENDER = ", reward_defender, "RFEWARD ATTACKER= ", reward_attacker)
    return played_cards, reward_attacker, reward_defender, output_defender, output_attacker, game_log