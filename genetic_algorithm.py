import sys
import time
import pickle

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model

sys.path.insert(0,'games') # to ensure game modules import correctly
from GwentLite import GwentLite

def generate_random_deck(config): # generate a random deck_list according to the game settings in config
    deck_list = []
    max_card_power = config['game_object'].max_card_power
    while len(deck_list) < config['game_object'].min_deck_size:
        if max_card_power == 1: deck_list.append(1) # if max_card_power is 1, that means we don't have any room to add cards above 1 power
        else:
            card = max( 1 , round( np.random.normal( config['game_object'].mean , np.random.choice(config['game_object'].stdev_range) ) ) ) # sample from normal distribution
            card = min( max_card_power , card ) # make sure the sample is greater or equal to 1 and less than or equal to max_card_power
            deck_list.append(card)
            max_card_power -= card - 1 # reduce max_card_power by one less of the card power, since max_card_power must always be at least 1
    deck_list[-1] += max_card_power - 1 # add the leftover max_card_power to the last card added to the deck_list
    return deck_list

def mutate_deck_list(deck_list,config): # mutate input deck_list according to config parameters
    deck_list_length = len(deck_list)
    
    num_cards_to_mutate = round( max( 1 , np.random.normal( config['mutation_params']['average_percentage_deck']*25 , config['mutation_params']['average_percentage_deck_range'] ) ) )
    indices_to_mutate = np.random.choice( range(deck_list_length) , size=num_cards_to_mutate , replace=False )

    new_deck_list = deck_list.copy()
    for index in indices_to_mutate: # mutate values
        value_shift = np.random.normal(0,config['mutation_params']['value_range'])
        value_shift = round(value_shift-0.5) if value_shift < 0 else round(value_shift+0.5) # increase value_shift by a magnitude of 0.5 in the direction of its sign, to prevent the sampled value_shift from being 0

        if new_deck_list[index] + value_shift < 1: new_deck_list[index] -= value_shift # change the value_shift from negative to positive and add it to the card power instead
        else: new_deck_list[index] += value_shift

    np.random.shuffle(new_deck_list)

    deck_power_difference = sum(new_deck_list) - config['game_object'].max_deck_power # adjust card values so that total deck power amounts to game.max_deck_power
    index = 0
    while deck_power_difference != 0:
        if deck_power_difference > 0 and new_deck_list[index] > 1: # we are over the game.max_deck_power constraint, so we must subtract
            new_deck_list[index] -= 1
            deck_power_difference -= 1
        else: # we are under the game.max_deck_power constraint, so we must add
            new_deck_list[index] += 1
            deck_power_difference += 1
        index = (index+1) % deck_list_length

    return new_deck_list

def play(deck_lists,model,config,human_player_index=None,print_game=True): # play a game with the two decklists, using model
    # deck_lists is a list[ deck0_list[int] , deck0_list[int] ]
    # human_player_index is the deck_list index that the human player will be using (None if you are using an AI to play both decklists against each other)
    game = config['game_object']

    # begin self play
    game.reset(deck_lists)
    state = game.get_features( game.get_player_turn() )
    while True:
        if print_game: print(game)
        
        player_index = game.get_player_turn()
        if player_index == human_player_index: action = int( input('Move: ') ) # human move
        else:
            action = np.argmax( model(state)[0,:] ) # AI move
            if print_game: print(f'==== AI model move: {action} ====')

        legal = game.act(action)
        if not legal: return {player_index:'illegal'}

        done,results_dict = game.check_game_over()

        if done: break

        state = game.get_features( game.get_player_turn() )
        
    if print_game: print(game)

    return results_dict

    

def genetic_algorithm(config):
    model = load_model(f"models/{config['game_object'].get_name()}/{str(config['model_name'])+'.h5'}")
    deck_pool = [ generate_random_deck(config) for _ in range(config['reproduction_params']['deck_pool_size']) ]

    start_time = time.time()
    for iteration_num in range( 1 , int(config['algorithm_config']['num_iterations'])+1 ):

        scores = np.array( [0 for _ in range(config['reproduction_params']['deck_pool_size'])] ) # +1 for win, 0 for tie, -1 for loss; no need to normalize or take average, since every deck plays the same amount of games
        for i in range(len(deck_pool)-1):
            for j in range(i+1,len(deck_pool)):

                for _ in range(config['algorithm_config']['num_test_games']): # play every decklist against each other num_test_games times
                    results_dict = play( (deck_pool[i],deck_pool[j]) , model , config , human_player_index=None , print_game=None )
                    
                    if 0 in results_dict: # player_index 0 corresponds to deck_pool[i]
                        if results_dict[0] == 'win': scores[i] += 1
                        elif results_dict[0] == 'loss': scores[i] -= 1
                    if 1 in results_dict: # player_index 1 corresponds to deck_pool[j]
                        if results_dict[1] == 'win': scores[j] += 1
                        elif results_dict[1] == 'loss': scores[j] -= 1
                    # if an illegal move is made, only the illegal player index will be in results_dict; in which case don't update either deck scores, since the model making an illegal move has no relation with the strength of a particular deck

        new_deck_pool = [] # get survivor decks and add them to new deck pool, along with their (mutated) offspring
        for deck_index in np.argsort(scores)[ -config['reproduction_params']['num_survivors'] : ] :
            surviving_deck = deck_pool[deck_index]
            new_deck_pool.append(surviving_deck)
            for _ in range(config['reproduction_params']['num_offspring']): new_deck_pool.append( mutate_deck_list(surviving_deck,config) )

        for _ in range( len(deck_pool) - len(new_deck_pool) ): new_deck_pool.append( generate_random_deck(config) ) # generate new random decks to fill in the remaining spaces in the deck_pool
        deck_pool = new_deck_pool

        if iteration_num % config['algorithm_config']['save_interval'] == 0:
            print(f'Iteration {iteration_num} complete - {time.time()-start_time} seconds')
            with open(f'games/gwent_lite_deck_lists/{iteration_num}.pkl','wb') as file: pickle.dump(deck_pool,file)
            start_time = time.time()
    
    return deck_pool

    

if __name__ == '__main__':
    game = GwentLite()
    
    config = {'game_object':game,
              'model_name':700000, # AI agent used to play and test deck_lists against each other
              'mutation_params': {'average_percentage_deck':0.1, # the average percentage of the deck that will be mutated; i.e. deck_size*average_percentage_deck cards will be mutated on average, every iteration
                                  'average_percentage_deck_range':1/game.min_deck_size, # the actual percentage for each mutation instance will be sampled from a normal distribution, with average_percentage_deck as mean and average_percentage_deck_range as stdev
                                  'value_range':1}, # a card selected to be mutated will have its power shifted by a value sampled from a normal distribution of mean 0 and stdev of value_range
              'reproduction_params': {'deck_pool_size':30, # the number of decks in the deck_pool per generation; the deck_pool is comprised of survivor decks and their offspring, while the remaining decks are new randomly generated decks
                                      'num_survivors':10, # take the top num_survivors best performing decks, to generate offspring for the next generation
                                      'num_offspring':1}, # number of offspring each surviving deck generates (using mutation_params); NOTE: the surviving deck itself is included in the next generation
              'algorithm_config': {'num_iterations':1e10, # number of iterations to run this algorithm
                                   'num_test_games':10, # number of games each deck in the deck_pool plays against every other deck
                                   'save_interval':20}, # save current deck_pool every save_interval iterations, and print time it took to execute save_interval iterations
              }

    with tf.device('/CPU:0'):
        m = genetic_algorithm(config)
    
