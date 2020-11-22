from Game import *
import numpy as np

class ConnectFour(Game):
    def __init__(self):
        super().__init__()
        self.board = np.zeros((2,6,7)) # playerX (6x7) , playerO (6x7)
        self.legal_moves = []
        self.player_turn = None
    def get_name(self): return 'Connect Four'
    def get_observation_shape(self): return 84
    def get_action_space_size(self): return 7
    def get_number_of_players(self): return 2
    def reset(self):
        self.board = np.zeros((2,6,7))
        self.legal_moves = [i for i in range(7)]
        self.player_turn = 0
    def get_player_turn(self): return self.player_turn
    def act(self,action):
        if action in self.legal_moves:
            summed_column = np.sum( self.board[:,:,action] , axis=0 )
            lowest_empty_row_index = np.where(summed_column==0)[0][-1]
            self.board[self.player_turn,lowest_empty_row_index,action] = 1
            if lowest_empty_row_index == 0: self.legal_moves.remove(action)
            self.player_turn = (self.player_turn+1) % 2
            return True # legal move
        return False # illegal move
    def check_game_over(self):
        for player_index in range(2):

            for row in range(6): # horizontal win
                for col in range(4):
                    if np.sum( self.board[player_index,row,col:col+4] )==4: return True,{player_index:'win',((player_index+1)%2):'loss'} # game_over,results_dict

            for col in range(7): # vertical win
                for row in range(3):
                    if np.sum( self.board[player_index,row:row+4,col] )==4: return True,{player_index:'win',((player_index+1)%2):'loss'}
                
            for row in range(3): # diagonal top left - bottom right win
                for col in range(4):
                    if np.sum( self.board[player_index][ [row+i for i in range(4)] , [col+i for i in range(4)] ] )==4: return True,{player_index:'win',((player_index+1)%2):'loss'}

            for row in range(3,6): # diagonal top left - bottom right win
                for col in range(4):
                    if np.sum( self.board[player_index][ [row-i for i in range(4)] , [col+i for i in range(4)] ] )==4: return True,{player_index:'win',((player_index+1)%2):'loss'}
                    
        if np.sum(self.board)==42: return True,{player_index:'tie',((player_index+1)%2):'tie'}
        return False,None # None indicates no winner yet
    def get_features(self,player_index): # 0 index of player_index is your board, and 1 index is opponent's board
        return self.board.copy()[[ player_index , (player_index+1)%2 ],:,:].reshape(1,-1)
    def sample_legal_move(self): return np.random.choice( self.legal_moves )
    def __str__(self):
        playerX = self.board[0,:,:].copy()
        playerO = self.board[1,:,:].copy()
        board = np.array( [ ['-' for _ in range(7)] for _ in range(6) ] )
        board[ playerX==1 ] = 'X'
        board[ playerO==1 ] = 'O'
        return str(board)
    def __repr__(self): return str(self)


if __name__ == '__main__':
    config = { 'self_play': {'num_episodes':410000,
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
    
    game = ConnectFour()
    game.play()

    
        
