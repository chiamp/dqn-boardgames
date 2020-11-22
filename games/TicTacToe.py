from Game import *
import numpy as np
from itertools import product

class TicTacToe(Game):
    def __init__(self):
        super().__init__()
        self.board = np.zeros((2,3,3)) # playerX (3x3) , playerO (3x3)
        self.legal_moves = {}
        self.player_turn = None
    def get_name(self): return 'Tic Tac Toe'
    def get_observation_shape(self): return 18
    def get_action_space_size(self): return 9
    def get_number_of_players(self): return 2
    def reset(self):
        self.board = np.zeros((2,3,3))
        self.legal_moves = {i:coord for i,coord in enumerate(product(range(3),range(3)))}
        self.player_turn = 0
    def get_player_turn(self): return self.player_turn
    def act(self,action):
        if action in self.legal_moves:
            self.board[self.player_turn][self.legal_moves[action]] = 1
            del self.legal_moves[action]
            self.player_turn = (self.player_turn+1) % 2
            return True # legal move
        return False # illegal move
    def check_game_over(self):
        for player_index in range(2):
            if np.sum(self.board[player_index,0,:])==3 or np.sum(self.board[player_index,1,:])==3 or np.sum(self.board[player_index,2,:])==3 or \
               np.sum(self.board[player_index,:,0])==3 or np.sum(self.board[player_index,:,1])==3 or np.sum(self.board[player_index,:,2])==3 or \
               np.sum(self.board[player_index,[0,1,2],[0,1,2]])==3 or np.sum(self.board[player_index,[0,1,2],[2,1,0]])==3:
                return True,{player_index:'win',((player_index+1)%2):'loss'} # game_over,results_dict
        if np.sum(self.board)==9: return True,{player_index:'tie',((player_index+1)%2):'tie'}
        return False,None # None indicates no winner yet
    def get_features(self,player_index): # 0 index of player_index is your board, and 1 index is opponent's board
        return self.board.copy()[[ player_index , (player_index+1)%2 ],:,:].reshape(1,-1)
    def sample_legal_move(self): return np.random.choice( list(self.legal_moves.keys() ) )
    def __str__(self):
        playerX = self.board[0,:,:].copy()
        playerO = self.board[1,:,:].copy()
        board = np.array( [ ['-' for _ in range(3)] for _ in range(3) ] )
        board[ playerX==1 ] = 'X'
        board[ playerO==1 ] = 'O'
        return str(board)
    def __repr__(self): return str(self)
    

if __name__ == '__main__':
    config = { 'self_play': {'num_episodes':30000,
                             'epsilon':0.1},
               'reward': {'win':1,
                          'tie':0,
                          'loss':-1,
                          'illegal_move':-1,
                          'discount_factor':0.9},
               'network': {'num_hidden_units':200,
                           'num_hidden_layers':3,
                           'hidden_activation':'relu',
                           'output_activation':'tanh',
                           'loss':'mse',
                           'learning_rate':1e-4},
               'training': {'replay_memory_capacity':10000,
                            'batch_size':1000,
                            'train_interval':10},
               'testing': {'test_interval':5000,
                           'num_test_episodes':100}
               }
    
    game = TicTacToe()
    game.play()
        
