import os
import sys
import time

import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential,clone_model,save_model,load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

sys.path.insert(0,'games') # to ensure game modules import correctly


def make_q_network(game,config):
    network_layers = [ Dense(config['network']['num_hidden_units'],activation=config['network']['hidden_activation'],
                             kernel_initializer='glorot_normal',input_shape=(game.get_observation_shape(),)) ]
    for _ in range(config['network']['num_hidden_layers']-1): network_layers.append( Dense(config['network']['num_hidden_units'],activation=config['network']['hidden_activation']) )
    network_layers.append( Dense(game.get_action_space_size(),activation=config['network']['output_activation'],kernel_initializer='glorot_normal') )
    q_network = Sequential(network_layers)
    q_network.compile(loss=config['network']['loss'],optimizer=Adam(learning_rate=config['network']['learning_rate']))
    return q_network
                                       
def q_learning(game,config):
    q_network_history = {}
    q_network = make_q_network(game,config)

    replay_memory = []

    start_time = time.time()
    for episode_num in range(1,int(config['self_play']['num_episodes'])+1):
        
        game.reset()

        state_histories = [ [] for _ in range(game.get_number_of_players()) ]
        action_histories = [ [] for _ in range(game.get_number_of_players()) ]
        
        state = game.get_features( game.get_player_turn() )
        while True:
            player_turn = game.get_player_turn()

            if np.random.sample() < config['self_play']['epsilon']: action = game.sample_legal_move()                    
            else: action = np.argmax( q_network(state)[0,:] )

            legal = game.act(action)
            
            new_state = game.get_features( game.get_player_turn() ) # in the perspective of opponent
            done,results_dict = game.check_game_over()

            state_histories[player_turn].append(state)
            action_histories[player_turn].append(action)

            if done or (not legal): break

            state = new_state

        if not legal: replay_memory.append( ( state_histories[player_turn][-1] , action_histories[player_turn][-1] , config['reward']['illegal_move'] ) )

        elif done:
            for player_index in range(game.get_number_of_players()):
                reward = config['reward'][results_dict[player_index]]

                expected_return = reward
                for i in range(len(state_histories[player_index])-1,-1,-1):
                    replay_memory.append( ( state_histories[player_index][i] , action_histories[player_index][i] , expected_return ) )
                    expected_return *= config['reward']['discount_factor']

        for _ in range( max( 0 , len(replay_memory) - config['training']['replay_memory_capacity'] ) ): replay_memory.pop(0)

        if episode_num % config['training']['train_interval'] == 0 and len(replay_memory) > config['training']['batch_size']:

            training_batch = np.vstack( random.sample(replay_memory,config['training']['batch_size']) )
            state_batch , action_batch , reward_batch = np.vstack(training_batch[:,0]) , training_batch[:,1] , training_batch[:,2]
            
            reward_matrix = q_network(state_batch).numpy()
            reward_matrix[ range(reward_matrix.shape[0]) , action_batch.astype(int) ] = reward_batch
            q_network.fit(state_batch, reward_matrix, epochs=1, verbose=0)

        if episode_num % config['testing']['test_interval'] == 0:
            print(f'\nIteration:{episode_num} - {time.time()-start_time} seconds')

            win,tie,loss,illegal,opponent_illegal = test(game,q_network,None,config['testing']['num_test_episodes'])
            print(f"vs Random\tWins:{win}/{config['testing']['num_test_episodes']} ({round(100*win/config['testing']['num_test_episodes'],2)}%)\tTies:{tie}/{config['testing']['num_test_episodes']} ({round(100*tie/config['testing']['num_test_episodes'],2)}%)\tLosses:{loss}/{config['testing']['num_test_episodes']} ({round(100*loss/config['testing']['num_test_episodes'],2)}%)\tIllegal moves:{illegal}/{config['testing']['num_test_episodes']} ({round(100*illegal/config['testing']['num_test_episodes'],2)}%)\tOpponent illegal moves:{opponent_illegal}/{config['testing']['num_test_episodes']} ({round(100*opponent_illegal/config['testing']['num_test_episodes'],2)}%)")
            
            for model_key in q_network_history:
                win,tie,loss,illegal,opponent_illegal = test(game,q_network,q_network_history[model_key],config['testing']['num_test_episodes'])
                print(f"vs {model_key}\tWins:{win}/{config['testing']['num_test_episodes']} ({round(100*win/config['testing']['num_test_episodes'],2)}%)\tTies:{tie}/{config['testing']['num_test_episodes']} ({round(100*tie/config['testing']['num_test_episodes'],2)}%)\tLosses:{loss}/{config['testing']['num_test_episodes']} ({round(100*loss/config['testing']['num_test_episodes'],2)}%)\tIllegal moves:{illegal}/{config['testing']['num_test_episodes']} ({round(100*illegal/config['testing']['num_test_episodes'],2)}%)\tOpponent illegal moves:{opponent_illegal}/{config['testing']['num_test_episodes']} ({round(100*opponent_illegal/config['testing']['num_test_episodes'],2)}%)")

            if not os.path.exists(f'models/{game.get_name()}'): os.mkdir(f'models/{game.get_name()}')
            save_model(q_network,f'models/{game.get_name()}/{episode_num}.h5')
            
            saved_network = clone_model(q_network)
            saved_network.set_weights(q_network.get_weights())
            q_network_history[episode_num] = saved_network

            start_time = time.time()

def test_one(game,current_q_network,opponent_q_network):
    player_index = np.random.choice(game.get_number_of_players())
    
    game.reset()
    state = game.get_features( game.get_player_turn() )
    while True:
        if player_index == game.get_player_turn(): action = np.argmax( current_q_network(state)[0,:] )
        elif opponent_q_network != None: action = np.argmax( opponent_q_network(state)[0,:] )
        else: action = game.sample_legal_move() # random sample action if opponent_q_network == None

        legal = game.act(action)
        if not legal:
            if player_index == game.get_player_turn(): return 'illegal' # player made an illegal move
            else: return 'opponent illegal' # opponent made an illegal move

        done,results_dict = game.check_game_over()

        if done: break

        state = game.get_features( game.get_player_turn() )

    return results_dict[player_index]
def test(game,current_q_network,opponent_q_network,n): # test current_q_network against another q_network
    test_results = []
    for _ in range(n): test_results.append( test_one(game,current_q_network,opponent_q_network) )
    return test_results.count('win') , test_results.count('tie') , test_results.count('loss') , test_results.count('illegal') , test_results.count('opponent illegal')

def play(game,model_name_list): # play against q_network
    
##    player_index = np.random.choice(game.get_number_of_players())
##    print(f'You are player {player_index}')

    player_indices = [i for i in range(game.get_number_of_players())]
    q_networks = {}
    for model_name in model_name_list:
        player_index = np.random.choice(player_indices,1)[0]
        q_networks[player_index] = load_model(f"models/{game.get_name()}/{str(model_name)+'.h5'}")
        player_indices.remove(player_index)

    print(f'Player indices:{player_indices}')
    print(f'AI model player indices:{list(q_networks.keys())}')
    
##    q_network = load_model(f"models/{game.get_name()}/{str(model_iteration_num)+'.h5'}")
##    q_networks = { ( (player_index+opponent_index+1) % game.get_number_of_players() ) : load_model(f"models/{game.get_name()}/{str(model_name)+'.h5'}") for opponent_index,model_name in enumerate(model_name_list) }
    
    game.reset()
    state = game.get_features( game.get_player_turn() )
    while True:
        print(game)
        if game.get_player_turn() in player_indices: action = int( input('Move: ') ) # action = game.sample_legal_move() # 
        else:
            action = np.argmax( q_networks[game.get_player_turn()](state)[0,:] )
            print(f'AI model move: {action}')

        legal = game.act(action)
        if not legal:
            if player_index == game.get_player_turn(): print('illegal') # player made an illegal move
            else: print('opponent illegal') # opponent made an illegal move
            return

        done,results_dict = game.check_game_over()

        if done: break

        state = game.get_features( game.get_player_turn() )
    print(game)

    print(results_dict)

def test_ai(game,q_network_list): # test q_networks against each other
    game.reset()
    state = game.get_features( game.get_player_turn() )
    while True:
        action = np.argmax( q_network_list[game.get_player_turn()](state)[0,:] )

        legal = game.act(action)
        if not legal:
            if player_index == game.get_player_turn(): print('illegal') # player made an illegal move
            else: print('opponent illegal') # opponent made an illegal move
            return

        done,results_dict = game.check_game_over()

        if done: break

        state = game.get_features( game.get_player_turn() )
    return results_dict
def test_ai_iter(game,models,n):
    stats = {}
    st = time.time()
    for _ in range(n):
        model_name_samples = random.sample(models.keys(),3)
        q_network_list = [models[model_name_key] for model_name_key in model_name_samples]
        results_dict = test_ai(game,q_network_list)
        for player_index in results_dict:
            model_name = model_name_samples[player_index]
            if model_name not in stats: stats[model_name] = {'win':0,'tie':0,'loss':0}
            stats[model_name][results_dict[player_index]] += 1
    print(f'{time.time()-st} seconds')
    model_name_list = []
    win_rate_list = []
    for model_name in stats.keys():
        win = stats[model_name]['win']
        tie = stats[model_name]['tie']
        loss = stats[model_name]['loss']
        total = win+tie+loss
        model_name_list.append(model_name)
        win_rate_list.append(win/total)
    for index in np.argsort(win_rate_list)[::-1]:
        model_name = model_name_list[index]
        win = stats[model_name]['win']
        tie = stats[model_name]['tie']
        loss = stats[model_name]['loss']
        total = win+tie+loss
        print(f'{model_name}\tWin:{win}/{total} ({round(100*win/total,2)}%)\tTie:{tie}/{total} ({round(100*tie/total,2)}%)\tLoss:{loss}/{total} ({round(100*loss/total,2)}%)')



if __name__ == '__main__':
    config = { 'self_play': {'num_episodes':1e10,
                             'epsilon':0.2},
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
               'testing': {'test_interval':25000,
                           'num_test_episodes':100}
               }
    
    from IncanGold import IncanGold
    game = IncanGold(using_other_game_interface=False)

##    with tf.device('/CPU:0'):
##        q_learning(game,config)
    
