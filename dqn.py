import os
import sys
import time

import random
from itertools import combinations
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential,clone_model,save_model,load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

sys.path.insert(0,'games') # to ensure game modules import correctly


########################################################################################################
########################################## LEARNING ALGORITHM ##########################################
########################################################################################################

class RandomNetwork: # "network" that plays only random moves
    def __init__(self,game): self.game = game
    def __call__(self,state):
        action = self.game.sample_legal_move()
        return np.array( [ [ 1 if action == i else 0 for i in range(game.get_action_space_size()) ] ] )
    
def make_q_network(game,config): # make a neural network according to config parameters
    network_layers = [ Dense(config['network']['num_hidden_units'],activation=config['network']['hidden_activation'],
                             kernel_initializer='glorot_normal',input_shape=(game.get_observation_shape(),)) ]
    for _ in range(config['network']['num_hidden_layers']-1): network_layers.append( Dense(config['network']['num_hidden_units'],activation=config['network']['hidden_activation']) )
    network_layers.append( Dense(game.get_action_space_size(),activation=config['network']['output_activation'],kernel_initializer='glorot_normal') )
    q_network = Sequential(network_layers)
    q_network.compile(loss=config['network']['loss'],optimizer=Adam(learning_rate=config['network']['learning_rate']))
    return q_network
                                       
def q_learning(game,config):
    # q_network_history stores previous network iterations, to be used when testing current network parameters against past parameters
    q_network_history = { 0:RandomNetwork(game) } # initialize with a random network at episode 0 to serve as a baseline when comparing future network performances
    q_network = make_q_network(game,config)

    replay_memory = [] # list[ tuple(state_t,action_t,reward_(t+1)) ]

    start_time = time.time()
    for episode_num in range(1,int(config['self_play']['num_episodes'])+1):
        
        game.reset()

        state_histories = [ [] for _ in range(game.get_number_of_players()) ] # list[ state_history_player0_list[state] , state_history_player1_list[state] , ... ]
        action_histories = [ [] for _ in range(game.get_number_of_players()) ] # list[ action_history_player0_list[action] , action_history_player0_list[action] , ... ]
        
        state = game.get_features( game.get_player_turn() )
        while True:
            player_turn = game.get_player_turn()

            if np.random.sample() < config['self_play']['epsilon']: action = game.sample_legal_move()                    
            else: action = np.argmax( q_network(state)[0,:] )

            legal = game.act(action)
            
            new_state = game.get_features( game.get_player_turn() ) # in the perspective of opponent
            done,results_dict = game.check_game_over() # results_dict is a dict{ player_index:'win'/'loss'/'tie' , ... }

            state_histories[player_turn].append(state)
            action_histories[player_turn].append(action)

            if done or (not legal): break

            state = new_state

        # only append the most recent state and action to the replay buffer if the episode terminated due to an illegal action
        if not legal: replay_memory.append( ( state_histories[player_turn][-1] , action_histories[player_turn][-1] , config['reward']['illegal_move'] ) )

        elif done: # otherwise, if the episode terminated naturally with a game over, calculate discounted rewards from the terminal reward and append all states and actions encountered to the replay buffer
            for player_index in range(game.get_number_of_players()):
                reward = config['reward'][results_dict[player_index]]

                expected_return = reward
                for i in range(len(state_histories[player_index])-1,-1,-1):
                    replay_memory.append( ( state_histories[player_index][i] , action_histories[player_index][i] , expected_return ) )
                    expected_return *= config['reward']['discount_factor']

        for _ in range( max( 0 , len(replay_memory) - config['training']['replay_memory_capacity'] ) ): replay_memory.pop(0) # remove the oldest replay samples if replay buffer is over capacity

        if episode_num % config['training']['train_interval'] == 0 and len(replay_memory) > config['training']['batch_size']: # train the network every set interval denoted by config

            training_batch = np.vstack( random.sample(replay_memory,config['training']['batch_size']) )
            state_batch , action_batch , reward_batch = np.vstack(training_batch[:,0]) , training_batch[:,1] , training_batch[:,2]
            # state_batch is a (batch_size x game_observation_shape) matrix , action_batch and reward_batch are (batch_size,) vectors
            
            reward_matrix = q_network(state_batch).numpy() # reward_matrix is a (batch_size x game_action_space_size) matrix, where each entry is the corresponding action value
            reward_matrix[ range(reward_matrix.shape[0]) , action_batch.astype(int) ] = reward_batch # assign the corresponding actions taken in the sample trajectories with the associated discounted reward
            q_network.fit(state_batch, reward_matrix, epochs=1, verbose=0)

        if episode_num % config['testing']['test_interval'] == 0:
            print(f'\nIteration {episode_num} complete - {time.time()-start_time} seconds')

            ordered_model_keys = list(q_network_history.keys()) # order the q_network episode_num keys in ascending order
            ordered_model_keys.sort()

            # test against all combinations of previous models against the current network's parameters
            for model_name_list in combinations( ordered_model_keys , game.get_number_of_players()-1 ):
                model_list = [q_network_history[model_name] for model_name in model_name_list] + [q_network] # for every possible combination of opponents, add our current q_network to the model_list
                test_ai( game , config['testing']['num_test_episodes'] , list(model_name_list) + [episode_num] , model_list ) # append current episode_num key to model_name_list and test games with the models in model_list
            
            if not os.path.exists(f'models/{game.get_name()}'): os.mkdir(f'models/{game.get_name()}') # create a subfolder for this game under the 'models' directory
            save_model(q_network,f'models/{game.get_name()}/{episode_num}.h5') # save the current model
            
            saved_network = clone_model(q_network) # add the current model to q_network_history
            saved_network.set_weights(q_network.get_weights())
            q_network_history[episode_num] = saved_network

            start_time = time.time()


#######################################################################################################
########################################## UTILITY FUNCTIONS ##########################################
#######################################################################################################
            
def play(game,model_name_list,model_list=[],print_game=True): # play the game against human(s) and/or q_network(s)
    # the number of human players is equal to ( game.get_number_of_players() - len(model_name_list) ), so you could play anything from an all human game, all the way to an all AI game
    # model_name_list: list[ model_name_strings ]
    # model_list: list[ keras_models_corresponding_to_model_name_strings_in_model_name_list ], if this is empty, then manually load models in using model filenames from model_name_list

    q_networks = {} # dict{ player_index:model }
    q_network_names = {} # dict{ player_index:model_name_string }
    
    player_indices = [i for i in range(game.get_number_of_players())] # initialize with all possible player indices, and then randomly take some away and assign them to AI models
    for i,model_name in enumerate(model_name_list): # randomly assign player indices for AI players
        player_index = np.random.choice(player_indices,1)[0] #i
        if len(model_list) > 0: q_networks[player_index] = model_list[i] # if model_list isn't empty, that means the models have already been instantiated
        else: q_networks[player_index] = load_model(f"models/{game.get_name()}/{str(model_name)+'.h5'}") # otherwise instantiate the models
        q_network_names[player_index] = model_name
        player_indices.remove(player_index)

    if print_game: print(f'Player indices:{player_indices}')
    if print_game: print(f'AI model player indices:{list(q_networks.keys())}')

    # begin self play
    game.reset()
    state = game.get_features( game.get_player_turn() )
    while True:
        if print_game: print(game)
        
        player_index = game.get_player_turn()
        if player_index in player_indices: action = int( input('Move: ') ) # human move
        else:
            action = np.argmax( q_networks[player_index](state)[0,:] ) # AI move
            if print_game: print(f'==== AI model move: {action} ====')

        legal = game.act(action)
        if not legal: return {player_index:'illegal'},q_network_names

        done,results_dict = game.check_game_over()

        if done: break

        state = game.get_features( game.get_player_turn() )
        
    if print_game: print(game)

    return results_dict,q_network_names

def test_ai(game,n,model_name_list=[],model_list=[]): # test AI models against each other
    # model_name_list: list[ model_name_strings ]
    # model_list: list[ keras_models_corresponding_to_model_name_strings_in_model_name_list ]
    # for every model in model_name_list/model_list, have them play n games against each other and compile winrate statistics
    # every combination of models from model_name_list/model_list are tested once per iteration of n
    
    # if model_name_list/model_list are empty, then load all models from the corresponding model directory for this game, and have them play against each other n times and compile winrate statistics
    if model_name_list == []: models_dict = { model_filename.split('.h5')[0] : load_model(f'models/{game.get_name()}/{model_filename}') for model_filename in os.listdir(f'models/{game.get_name()}') } # dict{ model_name:model }
    else: models_dict = { model_name_list[i] : model_list[i] for i in range(len(model_name_list)) }
    
    stats = {} # dict { model_name: {'win':int,'tie':int,'loss':int} }
    # get a list that contains all possible matchup combinations between all model AI's for this game
    model_combinations = [ model_name_list_subset for model_name_list_subset in combinations( models_dict.keys() , game.get_number_of_players() ) ]
    
    start_time = time.time()
    for _ in range(n):
        for model_name_list_subset in model_combinations: # model_name_list_subset is a subset of models to be tested in a game
            model_list_subset = [models_dict[model_name] for model_name in model_name_list_subset] # list of actual keras models, corresponding to model_name_list_subset
            results_dict,q_network_names = play(game,model_name_list_subset,model_list_subset,print_game=False) # play the game
            for player_index in q_network_names: # update the result of the game for every model that participated
                model_name = q_network_names[player_index] # player_indices are randomly assigned in game, so we need q_network_names to find the actual model_name
                if model_name not in stats: stats[model_name] = {'win':0,'tie':0,'loss':0,'illegal':0}
                # not all player indices will always be in results_dict if one player made an illegal move (in which case results_dict would only contain the index of the player that made that illegal move)
                if player_index in results_dict: stats[model_name][results_dict[player_index]] += 1
    print(f'Testing {n} games for models {list(models_dict.keys())} - {time.time()-start_time} seconds')
    
    ordered_model_name_list = [] # keep track of model names
    win_rate_list = [] # indices correspond to model indices in ordered_model_name_list
    for model_name in models_dict.keys(): # populate the two lists
        ordered_model_name_list.append(model_name)
        win_rate_list.append( stats[model_name]['win'] / max( 1 , (stats[model_name]['win']+stats[model_name]['tie']+stats[model_name]['loss']+stats[model_name]['illegal']) ) ) # max to prevent 0 division error
    for index in np.argsort(win_rate_list)[::-1]: # print results for each model in descending order according to winrate
        model_name = ordered_model_name_list[index]
        win,tie,loss,illegal = stats[model_name]['win'],stats[model_name]['tie'],stats[model_name]['loss'],stats[model_name]['illegal']
        total = win+tie+loss+illegal
        print(f'{model_name}\tWin:{win}/{total} ({round(100*win/max(1,total),2)}%)\tTie:{tie}/{total} ({round(100*tie/max(1,total),2)}%)\tLoss:{loss}/{total} ({round(100*loss/max(1,total),2)}%)\tIllegal:{illegal}/{total} ({round(100*illegal/max(1,total),2)}%)')



if __name__ == '__main__':
    config = { 'self_play': {'num_episodes':1e10,
                             'epsilon':0.1},
               'reward': {'win':1,
                          'tie':0,
                          'loss':-1,
                          'illegal_move':-1,
                          'discount_factor':1.},
               'network': {'num_hidden_units':200,
                           'num_hidden_layers':3,
                           'hidden_activation':'relu',
                           'output_activation':'tanh',
                           'loss':'mse',
                           'learning_rate':1e-4},
               'training': {'replay_memory_capacity':10000,
                            'batch_size':1000,
                            'train_interval':10},
               'testing': {'test_interval':100000,
                           'num_test_episodes':10}
               }
    
    from GwentLite import GwentLite
    game = GwentLite()
##    play(game,[700000])

##    with tf.device('/CPU:0'):
##    q_learning(game,config)

##    with tf.device('/CPU:0'):
##        test_ai(game,11446,model_name_list=[],model_list=[])
    
