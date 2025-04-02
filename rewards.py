import torch

def rewards(game_log,
            game,
            reward_value,
            defence_decision,
            defender_card_prob,
            attacker_card_prob,
            mean_masked_attacker_action_probs,
            mean_masked_defender_action_probs,
            episode,
            step_number,
            chosen_attackers_card,
            chosen_defender_card,
            attacker,
            defender,
            not_playing_cards,
            reward_attacker,
            reward_defender,
            played_cards,
            done,
            counter,
            probability_to_defend,
            decision_to_continue_attack,
            verbose = False):


    #print("mean_masked_attacker_action_probs = ", mean_masked_attacker_action_probs)
    
    result = ""
    if defence_decision == "failure":
        if verbose: print("Defence Failure")
        result = "Wrong card chosen"
        done = True
        reward_defender = reward_defender - reward_value * defender_card_prob * (probability_to_defend +0.5 )* 2  # Penalize defender for failing defense

    # elif defence_decision == "withdraw":
    #     result = "Attacker wins"
    #     done = True
    #     reward_defender = reward_defender - reward_value * normalized_defender_card_prob * 0.5  # Penalize defender for withdrawing

    if chosen_defender_card not in game.players[defender] or not game.can_beat(chosen_attackers_card, chosen_defender_card):
        done = True
        result = "Attacker wins"
        chosen_defender_card = None
        if verbose: print("Attacker wins from the first turn")
    else:
        game.players[defender].remove(chosen_defender_card)

        # No more cards in hand remaining, defender / attacker wins
        if not game.players[attacker]:
            done = True
            result = "Attacker wins, no cards remaining"
            played_cards = not_playing_cards.clone()
        elif not game.players[defender]:
            done = True
            result = "Defender wins, no cards remaining"
            played_cards = not_playing_cards.clone()
        elif not game.players[defender] and not game.players[attacker]:
            done = True
            result = "No winner, no cards remaining"
            played_cards = not_playing_cards.clone()
        else:
            result = "Defender wins"

    if result == "Attacker wins, no cards remaining" or result == "Defender wins":
        reward_defender = reward_defender + reward_value * defender_card_prob

    if result == "Attacker wins" or result == "Attacker wins, no cards remaining":
        reward_attacker = reward_attacker + reward_value * attacker_card_prob

    if result == "Attacker wins":
        reward_attacker = reward_attacker + reward_value  + reward_value * attacker_card_prob # Reward attacker for winning the set

    if result == "Defender wins":
        reward_defender = reward_defender + reward_value + reward_value * defender_card_prob # Reward defender for winning the set

    # Penalize attacker for refusing to attack with valid cards
    if defence_decision == "withdraw" and any(card[0] == chosen_attackers_card[0] for card in game.players[attacker]):
        reward_attacker = reward_attacker - reward_value * attacker_card_prob

    # Penalize defender for refusing to defend with valid cards
    if defence_decision == "withdraw" and any(game.can_beat(chosen_attackers_card, card) for card in game.players[defender]):
        reward_defender = reward_defender - reward_value * defender_card_prob 



    if torch.isnan(mean_masked_attacker_action_probs):
        mean_masked_attacker_action_probs = torch.tensor([1.], dtype=torch.float32, requires_grad=True)

    if torch.isnan(mean_masked_defender_action_probs):
        mean_masked_defender_action_probs = torch.tensor([1.], dtype=torch.float32, requires_grad=True)


    if verbose: print(f"Episode {episode + 1}:" + result)
    game_log.append({
        'episode': episode + 1,
        'step': counter,
        'trump': game.trump,
        'attacker_action': chosen_attackers_card,
        'defender_action': chosen_defender_card,
        'attacker_card_prob': attacker_card_prob.detach().numpy(),
        'defender_card_prob': defender_card_prob.detach().numpy(),
        'mean_masked_attacker_action_probs': mean_masked_attacker_action_probs.detach().numpy(),
        'mean_masked_defender_action_probs': mean_masked_defender_action_probs.detach().numpy(),
        'remaining_attacker_hand': list(game.players[attacker]),
        'remaining_defender_hand': list(game.players[defender]),
        'result': result
    })
    
    
    
    reward_attacker = reward_attacker - 1*torch.log(attacker_card_prob)  + 1.* torch.log( 1 + mean_masked_attacker_action_probs )
    reward_defender = reward_defender - 1*torch.log(defender_card_prob) + 1.* torch.log( 1 + mean_masked_defender_action_probs )
    

    
    return done, played_cards, reward_attacker, reward_defender, game_log, defence_decision 
