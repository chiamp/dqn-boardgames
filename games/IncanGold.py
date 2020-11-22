from Game import *
import numpy as np


TREASURE_VALUE_ONEHOT_DECK_MAPPING = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 7: 5, 9: 6, 11: 7, 13: 8, 14: 9, 15: 10} # (treasure_value) --> (deck_onehot_index)

class Card:
    def __init__(self,treasure_value,is_artifact,hazard_id):
        self.treasure_value = treasure_value
        self.is_artifact = is_artifact
        self.hazard_id = hazard_id
        self.deck_onehot_index = (TREASURE_VALUE_ONEHOT_DECK_MAPPING[self.treasure_value] if self.treasure_value != None else 0) + (11+self.hazard_id if self.hazard_id != None else 0) + (16 if self.is_artifact else 0)
        self.feature_representation = np.array( [ treasure_value/15 if treasure_value != None else 0 ] + [ 1 if is_artifact else 0 ] + [ 1 if i == hazard_id else 0 for i in range(5) ] , dtype=float )
    def get_features(self): return self.feature_representation
    def __str__(self): return (f'T{self.treasure_value}' if self.treasure_value != None else '') + ('A' if self.is_artifact else '') + (f'H{self.hazard_id}' if self.hazard_id != None else '')
    def __repr__(self): return str(self)

ARTIFACT_CARD = Card(None,True,None) # static reference to artifact card
HAZARD_LIST = ['fire', 'mummy', 'rocks', 'snakes', 'spiders']

class Deck:
    def __init__(self):
        self.deck = [] # last index is the top of the pile
        self.deck_size = 0
        self.feature_representation = np.array([0 for _ in range(17)],dtype=float) # 11-tuple(treasure_cards) + 5-tuple(hazard_cards) + artifact_count

        # constant/static attributes
        self.DECK_CARDS = [ Card(treasure_value,False,None) for treasure_value in [1, 2, 3, 4, 5, 5, 7, 7, 9, 11, 11, 13, 14, 15] ]
        for hazard_id in range(5):
            for _ in range(3): self.DECK_CARDS.append(Card(None,False,hazard_id))
    def reset(self):
        self.deck.clear()
        self.feature_representation *= 0
        for card in self.DECK_CARDS:
            self.deck.append(card)
            self.feature_representation[card.deck_onehot_index] += 1
        self.deck.append(ARTIFACT_CARD)
        self.feature_representation[ARTIFACT_CARD.deck_onehot_index] += 1
        self.deck_size = 30
        np.random.shuffle(self.deck)
    def draw(self):
        card = self.deck.pop(-1)
        self.deck_size -= 1
        self.feature_representation[card.deck_onehot_index] -= 1
        return card
    def draw_specific(self): # a function to be used when playing with another game interface
        # input what card was drawn in that game interface
        # to be used for testing AI models against other players using another game interface
        card_string = input('Card drawn ( t[1-15] , a , [fire, mummy, rocks, snakes, spiders] ): ') # card format should match the string representation of Card class
        treasure_value = None
        is_artifact = False
        hazard_id = None
        while True:
            if card_string[0] == 't':
                treasure_value = int(card_string[1:])
                break
            elif card_string == 'a':
                is_artifact = True
                break
            elif card_string in HAZARD_LIST:
                hazard_id = HAZARD_LIST.index(card_string)
                break
            else: card_string = input('Incorrect card string format\nCard drawn ( t[1-15] , a , [fire, mummy, rocks, snakes, spiders] ): ')
        found_matching_card = False
        for card in self.deck:
            if card.treasure_value == treasure_value and card.is_artifact == is_artifact and card.hazard_id == hazard_id:
                found_matching_card = True
                self.deck.remove(card)
                break
        if not found_matching_card:
            print('Card not found in deck')
            raise Exception
        self.deck_size -= 1
        self.feature_representation[card.deck_onehot_index] -= 1
        return card
    def add(self,card_list):
        for card in card_list:
            self.deck.append(card)
            self.deck_size += 1
            self.feature_representation[card.deck_onehot_index] += 1
        np.random.shuffle(self.deck)
    def get_features(self): return self.feature_representation / self.deck_size
    def __str__(self): return str(self.deck)
    def __repr__(self): return str(self)

class Board:
    def __init__(self):
        self.board = []
        self.board_points = 0
        self.num_5_point_artifacts = 0
        self.num_10_point_artifacts = 0
        self.num_artifacts_retrieved = 0
        self.hazard_count = {hazard_id:0 for hazard_id in range(5)}
        self.num_active_players = 0
        self.feature_representation = np.array( [0 for _ in range(9) ] , dtype=float ) # self.board_points , self.num_5_point_artifacts , self.num_10_point_artifacts , 5-tuple(hazard count) , self.num_active_players
        # constant/static attribute
        self.NORMALIZATION_FACTOR = np.array( [17,3,2,1,1,1,1,1,3] , dtype=float ) # 3.1379029954958635 3.138725295488719 17
    def reset(self): # same as next_round method but reset self.num_artifacts_retrieved()
        self.next_round()
        self.num_artifacts_retrieved = 0
    def next_round(self): # reset everything except self.num_artifacts_retrieved (since we need to keep track of this for calculating points earned from artifacts)
        self.board.clear()
        self.board_points = 0
        self.num_5_point_artifacts = 0
        self.num_10_point_artifacts = 0
        for hazard_id in self.hazard_count: self.hazard_count[hazard_id] = 0
        self.num_active_players = 3
        self.feature_representation *= 0
        self.feature_representation[8] = 3
    def add(self,card): # return tuple(boolean,int), indicating whether board has two hazard cards of the same type (round_over) and how many points each active player gets this round, to be added to IncanGold.player_board_points
        self.board.append(card)
        if card.treasure_value != None:
            added_points = card.treasure_value % self.num_active_players
            self.board_points += added_points
            self.feature_representation[0] += added_points
            return False , card.treasure_value // self.num_active_players
        elif card.is_artifact:
            if self.num_artifacts_retrieved < 3:
                self.num_5_point_artifacts += 1
                self.feature_representation[1] += 1
            else:
                self.num_10_point_artifacts += 1
                self.feature_representation[2] += 1
            return False , 0
        else: # hazard card
            if self.hazard_count[card.hazard_id] == 1: return True , 0
            self.hazard_count[card.hazard_id] += 1
            self.feature_representation[3+card.hazard_id] += 1
            return False , 0
    def leave(self,num_leavers): # return int indicating how many points each leaving player gets
        if num_leavers == 0: return 0
        points_taken = self.board_points // num_leavers
        self.board_points -= points_taken*num_leavers
        self.feature_representation[0] -= points_taken*num_leavers
        self.num_active_players -= num_leavers
        self.feature_representation[8] -= num_leavers
        if num_leavers == 1:
            for artifact in range(self.num_5_point_artifacts):
                points_taken += 5
                self.num_5_point_artifacts -= 1
                self.feature_representation[1] -= 1
                self.num_artifacts_retrieved += 1
            for artifact in range(self.num_10_point_artifacts):
                points_taken += 10
                self.num_10_point_artifacts -= 1
                self.feature_representation[2] -= 1
                self.num_artifacts_retrieved += 1
            while ARTIFACT_CARD in self.board: self.board.remove(ARTIFACT_CARD)
        return points_taken
    def get_features(self): return self.feature_representation / self.NORMALIZATION_FACTOR
    def __str__(self): return f'Board: {str(self.board)}\nBoard points: {self.board_points}\nNumber of active players: {self.num_active_players}'
    def __repr__(self): return str(self)
            
class IncanGold(Game):
    def __init__(self,using_other_game_interface=False):
        super().__init__()
        self.deck = Deck()
        self.deck_draw_function = self.deck.draw_specific if using_other_game_interface else self.deck.draw
        self.board = Board()
        self.round = 0
        self.player_points = {0:0,1:0,2:0}
        self.player_board_points = {0:0,1:0,2:0} # temporary points that players have on the board; they must leave the temple to cash in their points
        self.total_points = 0 # normalization factor for self.player_points
        self.active_player_index = 0 # indexed for self.active_players, NOT player_index itself
        self.active_players = [] # list[ player_indices who are still in this round ]
        self.player_actions = {} # buffer to hold player actions, and then execute them simultaneously (since this game isn't turn based)
    def get_name(self): return 'Incan Gold'
    def get_observation_shape(self): return 34
    def get_action_space_size(self): return 2
    def get_number_of_players(self): return 3
    def reset(self):
        self.deck.reset()
        self.board.reset()
        
        self.round = 1 # 1-indexed
        
        for player_index in self.player_points:
            self.player_points[player_index] = 0
            self.player_board_points[player_index] = 0
        self.total_points = 0
        self.active_player_index = 0
        self.active_players = [player_index for player_index in range(3)]
        self.player_actions.clear()

        two_hazards_found,points_earned = self.board.add( self.deck_draw_function() ) # start the round with a card already on the board
        for player_index in self.active_players: self.player_board_points[player_index] += points_earned # they don't cash in until they leave
    def next_round(self):
        while ARTIFACT_CARD in self.board.board: self.board.board.remove(ARTIFACT_CARD) # clear all artifacts that were on the board (they are lost forever)
        self.deck.add(self.board.board + [ARTIFACT_CARD]) # add cards on board back to deck and shuffle a new artifact card in
        np.random.shuffle(self.deck.deck)
        self.board.next_round()
        
        self.round += 1 # 1-indexed
        
        for player_index in self.player_board_points: self.player_board_points[player_index] = 0
        self.active_player_index = 0
        self.active_players = [player_index for player_index in range(3)]
        self.player_actions.clear()

        two_hazards_found,points_earned = self.board.add( self.deck_draw_function() ) # start the round with a card already on the board
        for player_index in self.active_players: self.player_board_points[player_index] += points_earned # they don't cash in until they leave
    def get_player_turn(self): return self.active_players[self.active_player_index] # return player index of active player
    def act(self,action): # 0 leave , 1 stay; both actions are always legal if you're an active player
        self.player_actions[ self.active_players[self.active_player_index] ] = action # add action to self.player_actions buffer and wait until it is fully filled before executing actions

        if len(self.player_actions) == len(self.active_players): # execute actions
            leavers = []
            for player_index in self.player_actions:
                if self.player_actions[player_index] == 0:
                    leavers.append(player_index)
                    self.active_players.remove(player_index)

            points_taken = self.board.leave(len(leavers)) # update self.board
            for leaver_index in leavers:
                total_points_taken = self.player_board_points[leaver_index] + points_taken
                self.player_points[leaver_index] += total_points_taken # update leaver self.player_points
                self.total_points += total_points_taken # update self.total_points
                self.player_board_points[leaver_index] = 0

            if len(self.active_players) == 0: # round over, go to next round
                self.next_round()
                return True

            two_hazards_found,points_earned = self.board.add( self.deck_draw_function() ) # only add a new card if there's at least one active player
            if not two_hazards_found:
                for player_index in self.active_players: self.player_board_points[player_index] += points_earned # they don't cash in until they leave
            elif two_hazards_found: # round over, take out the last hazard card, and go to next round
                self.board.board.pop(-1)
                self.next_round()
                return True
            
            self.player_actions.clear() # clear action buffer for next turn

            self.active_player_index = 0 # go back to the earliest player's turn after executing all actions
        
        else: self.active_player_index = (self.active_player_index+1) % len(self.active_players) # otherwise increment player turn
        
        return True
    def check_game_over(self):
        results_dict = {}
        if self.round > 5:
            scores = [self.player_points[player_index] for player_index in range(3)]
            third_place_index,second_place_index,first_place_index = np.argsort(scores)
            if scores[first_place_index] == scores[second_place_index]:
                results_dict[first_place_index] = 'tie'
                results_dict[second_place_index] = 'tie'
                if scores[third_place_index] == scores[first_place_index]: results_dict[third_place_index] = 'tie'
                else: results_dict[third_place_index] = 'loss'
            else:
                results_dict[first_place_index] = 'win'
                results_dict[second_place_index] = 'loss'
                results_dict[third_place_index] = 'loss'
            return True,results_dict
        return False,None
    def get_features(self,player_index):
        # Player: ( player_points , player_board_points )
        # Opponent (highest): ( player_points , player_board_points )
        # Opponent (lower): ( player_points , player_board_points )
        # Which opponent(s) are active: Binary 2-tuple([0/1,0/1]), # corresponding to the order opponents with highest points first, lower second
        # Board: board_points , num_5_point_artifacts , num_10_point_artifacts , 5-tuple(hazard count) , num_active_players
        # Deck: Onehot(17)
        
        total_player_and_player_board_points = self.total_points
        for p_index in self.active_players: total_player_and_player_board_points += self.player_board_points[p_index]
        if total_player_and_player_board_points == 0: total_player_and_player_board_points = 1
        
        opponent_indices = [ opponent_index for opponent_index in range(3) if opponent_index != player_index ]
        lower_opponent_index,highest_opponent_index = np.argsort( [ self.player_points[opponent_index] for opponent_index in opponent_indices ] )
        lower_opponent_index,highest_opponent_index = opponent_indices[lower_opponent_index],opponent_indices[highest_opponent_index]

##        print(f'Player: {self.player_points[player_index] / total_player_and_player_board_points , self.player_board_points[player_index] / total_player_and_player_board_points}')
##        print(f'Opponent (highest): {self.player_points[highest_opponent_index] / total_player_and_player_board_points , self.player_board_points[highest_opponent_index] / total_player_and_player_board_points}')
##        print(f'Opponent (lower): {self.player_points[lower_opponent_index] / total_player_and_player_board_points , self.player_board_points[lower_opponent_index] / total_player_and_player_board_points}')
##        print(f'Opponent active: {1 if highest_opponent_index in self.active_players else 0 , 1 if lower_opponent_index in self.active_players else 0}')
##        print(f'Board: {self.board.get_features()}')
##        print(f'Deck: {self.deck.get_features()}')
        
        return np.concatenate( ( [ self.player_points[player_index] / total_player_and_player_board_points , self.player_board_points[player_index] / total_player_and_player_board_points ,
                                   self.player_points[highest_opponent_index] / total_player_and_player_board_points , self.player_board_points[highest_opponent_index] / total_player_and_player_board_points ,
                                   self.player_points[lower_opponent_index] / total_player_and_player_board_points , self.player_board_points[lower_opponent_index] / total_player_and_player_board_points ,
                                   1 if highest_opponent_index in self.active_players else 0 , 1 if lower_opponent_index in self.active_players else 0 ] ,
                                 self.board.get_features() , self.deck.get_features() ) ).reshape(1,-1)
    def sample_legal_move(self): return np.random.choice(2)
    def __str__(self):
        deck_string = f'\n{str(self.deck)}\n'
        board_string = f'{str(self.board)}\n\n'
        round_info_string = f'Round: {self.round}\tActive player: {self.active_players[self.active_player_index]}\n'
        player_strings = '\n'.join( [ f'Player {player_index}\tPoints:{self.player_points[player_index]}\tBoard Points:{self.player_board_points[player_index]}\tActive: {True if player_index in self.active_players else False}' for player_index in range(3) ] )
        return board_string + round_info_string + player_strings # deck_string + 
    def __repr__(self): return str(self)
    def play(self): # play a human vs human game
        while True:
            self.reset()

            game_over,winner = self.check_game_over()
            while not game_over:
                print(self)
##                self.get_features(self.get_player_turn())
                a = int( input('Move: ') )
                while not self.act(a): a = int( input('Illegal!\nMove: ') )
                game_over,winner = self.check_game_over()
            print(self)
##            self.get_features(self.get_player_turn())

            print(self.check_game_over())


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
    
    game = IncanGold(using_other_game_interface=True)
    game.play()
