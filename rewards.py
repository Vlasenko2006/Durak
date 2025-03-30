def rewards(game_log,
            game,
            reward_value,
            defence_decision,
            defender_card_prob,
            attacker_card_prob,
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
            verbose = False):
    
    result = ""
    if defence_decision == "failure":
        if verbose: print("Defence Failure")
        result = "Wrong card chosen"
        done = True
        reward_defender -= reward_value  # Penalize defender for failing defense

    elif defence_decision == "withdraw":
        result = "Attacker wins"
        done = True
        reward_defender -= reward_value  # Penalize defender for withdrawing

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
        reward_attacker += reward_value  # Reward attacker for winning the set

    if result == "Defender wins":
        reward_defender += reward_value  # Reward defender for winning the set

    # Penalize attacker for refusing to attack with valid cards
    if defence_decision == "withdraw" and any(card[0] == chosen_attackers_card[0] for card in game.players[attacker]):
        reward_attacker -= reward_value

    # Penalize defender for refusing to defend with valid cards
    if defence_decision == "withdraw" and any(game.can_beat(chosen_attackers_card, card) for card in game.players[defender]):
        reward_defender -= reward_value

    if verbose: print(f"Episode {episode + 1}:" + result)
    game_log.append({
        'episode': episode + 1,
        'step': counter,
        'trump': game.trump,
        'attacker_action': chosen_attackers_card,
        'defender_action': chosen_defender_card,
        'remaining_attacker_hand': list(game.players[attacker]),
        'remaining_defender_hand': list(game.players[defender]),
        'result': result
    })

    return done, played_cards, reward_attacker, reward_defender, game_log
