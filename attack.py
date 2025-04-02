import torch
from check_combinations import mask_invalid_cards


def attack(attacker_net, 
           attacker,
           attack_value, 
           game,
           attack_flag,
           played_cards,
           cards_on_a_table,
           full_deck, 
           episode, 
           verbose = False
           ):
    
    # Attacker's turn
    
    # print("Attack")
    # print('========================================\n')
    
    
    done = False

    # print(f"Debug: Starting attack function for attacker {attacker}")
    # print(f"Debug: Initial cards_on_a_table = {cards_on_a_table}")
    
    # Get attackers cards (suit,value)     
    valid_attacker_cards = [card for card in game.players[attacker] if card in full_deck]
    
    # attack_value is a card chosen for attack (provided that one attack was already done)
    # here we ensure that the attacker can only play cards that match the ongoing attack value, 
    # maintaining the rules of the game. To prevent the case that NN chooses the card it does not possess
    if attack_value:
        valid_attacker_cards = [card for card in valid_attacker_cards if card[0] == attack_value]
   
    if not valid_attacker_cards:
        done = True
        if verbose:  print(f"Episode {episode + 1}: No valid cards to attack.")
    
    empty_index = torch.tensor([[-1]], dtype=torch.float32)  # this index specifies that we will attack
    state_attacker = torch.tensor(game.get_state(attacker), dtype=torch.float32, requires_grad=True).unsqueeze(0)
    # print(f"Debug: Initial state_attacker = {state_attacker}")

    output_attacker = attacker_net(state_attacker,
                                   attack_flag, 
                                   played_cards, 
                                   cards_on_a_table,
                                   empty_index)
    
    attacker_action_probs = output_attacker[..., :-1]
    decision_to_continue_attack = output_attacker[..., -1]

    # print(f"Debug: Valid attacker card: {valid_attacker_cards}")
    masked_attacker_action_probs, mask = mask_invalid_cards(attacker_action_probs,
                                                            valid_attacker_cards,
                                                            full_deck)
    
    attacker_card_index = torch.argmax(masked_attacker_action_probs).item()
    
    chosen_attackers_card = game.index_to_card(attacker_card_index)
    
    #print(f"Debug: Chosen attacker card: {chosen_attackers_card}")
    attacker_card_prob = masked_attacker_action_probs[0, attacker_card_index]
    # print(f"Debug: Chosen attacker card: {chosen_attackers_card}")
    # Compute mean of masked_attacker_action_probs excluding attacker_card_index
    masked_probs_excluding_index = masked_attacker_action_probs.clone()
    masked_probs_excluding_index[0, attacker_card_index] = 0  # Set the value at attacker_card_index to 0
    mean_masked_attacker_action_probs = masked_probs_excluding_index[masked_probs_excluding_index != 0].mean()

    cards_on_a_table = game.update_state(attacker,
                                         attacker_card_index,
                                         cards_on_a_table)
    # print(f"Debug: Updated cards_on_a_table = {cards_on_a_table}")

   # print(f"Debug: game.players[attacker] before removing chosen card: {game.players[attacker]}")
    if chosen_attackers_card not in game.players[attacker]:
        # Invalid move, attacker loses
        done = True

        if attack_value is None:
            attack_value = chosen_attackers_card[0]
            
    game.players[attacker].remove(chosen_attackers_card)     
    # print(f"Debug: game.players[attacker] after removing chosen card: {game.players[attacker]}", "LEN = ", len(game.players[attacker]))
    # print("\n")
    
    
    # print("Attak has been finished", cards_on_a_table)
    # print('========================================\n')
    
    
    return decision_to_continue_attack, attacker_card_prob, chosen_attackers_card, \
        attacker_card_index, cards_on_a_table, done, output_attacker, mean_masked_attacker_action_probs