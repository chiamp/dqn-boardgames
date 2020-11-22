# abstract game class
class Game:
    def __init__(self): pass
    def get_name(self): pass # return str
    def get_observation_shape(self): pass # return int
    def get_action_space_size(self): pass # return int
    def get_number_of_players(self): pass # return int
    def reset(self): pass # reset game, return None
    def get_player_turn(self): pass# return int
    def act(self,action): pass # apply (int) action to game state, return boolean indicating whether action was legal or not
    def check_game_over(self): pass # check whether game is over, return a tuple(bool,dict) that indicates if the game is over and the result dict{player_index:result('win'/'tie'/'loss'). If False, then winner doesn't exist, so return None.
    def get_features(self,player_index): pass # return features of game state
    def sample_legal_move(self): pass # sample a legal move from current game state
    def __str__(self): pass # string representation of game state
    def __repr__(self): pass # return string method
    def play(self): # play a human vs human game
        while True:
            self.reset()

            game_over,winner = self.check_game_over()
            while not game_over:
                print(self)
                a = int( input('Move: ') )
                while not self.act(a): a = int( input('Illegal!\nMove: ') )
                game_over,winner = self.check_game_over()
            print(self)

            print(self.check_game_over())
