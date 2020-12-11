from Game import *
import numpy as np

class Deck:
    def __init__(self,min_deck_size,max_deck_power):
        self.min_deck_size = min_deck_size # the minimum amount of cards required in a player deck
        self.max_deck_power = max_deck_power # the maximum total sum of a deck
        self.max_card_power = self.max_deck_power - self.min_deck_size + 1 # the maximum power value a card can have given self.min_deck_size and self.max_deck_power (i.e. if all other cards were power 1)
        
        self.deck = [] # last index is the top of the pile
        self.deck_size = 0
        self.feature_representation = np.array([0 for _ in range(self.max_card_power)],dtype=float) # onehot for every unique card power; value represents the count for that specific card index
    def reset(self,input_deck_list,mean=None,stdev=None): # reset the deck
        # if input_deck_list is a list (not None), set the deck to the input_deck_list
        # otherwise generate a random deck_list by sampling cards from a normal distribution using mean and stdev
        
        if input_deck_list != None: deck_list = input_deck_list.copy()
        else:
            deck_list = []
            max_card_power = self.max_card_power
            while len(deck_list) < self.min_deck_size:
                if max_card_power == 1: deck_list.append(1) # if max_card_power is 1, that means we don't have any room to add cards above 1 power
                else:
                    card = max( 1 , round( np.random.normal(mean,stdev) ) ) # sample from normal distribution
                    card = min( max_card_power , card ) # make sure the sample is greater or equal to 1 and less than or equal to max_card_power
                    deck_list.append(card)
                    max_card_power -= card - 1 # reduce max_card_power by one less of the card power, since max_card_power must always be at least 1
            deck_list[-1] += max_card_power - 1 # add the leftover max_card_power to the last card added to the deck_list

        self.deck = deck_list
        self.deck_size = len(deck_list)
        self.feature_representation *= 0
        for card in self.deck: self.feature_representation[card-1] += 1 # cards are integers that go from 1 to self.max_card_power, but self.feature_representation is 0-indexed
        np.random.shuffle(self.deck) 
    def draw(self):
        card = self.deck.pop(-1)
        self.deck_size -= 1
        self.feature_representation[card-1] -= 1
        return card
    def get_features(self): return self.feature_representation / self.deck_size
    def __str__(self): return str(self.deck)
    def __repr__(self): return str(self)

# abstract game class
class GwentLite(Game):
    def __init__(self):
        # static game settings
        self.min_deck_size = 25 # the minimum amount of cards required in a player deck
        self.max_deck_power = 100 # the maximum total sum of a deck
        self.max_card_power = self.max_deck_power - self.min_deck_size + 1 # the maximum power value a card can have given self.min_deck_size and self.max_deck_power (i.e. if all other cards were power 1)
        self.mean = round(self.max_deck_power/self.min_deck_size) #range(1,int(self.max_card_power/10)+1) # mean param for normal distribution when generating random decks
        # average power needed to win a round = average power per card, multiplied by 16 (number of cards you can play per game), divided by 3 rounds
        # maximum power per card = average power needed to win a round divided by 3 cards, since you draw 3 new cards at the start of each round
        self.stdev_range = range( 0 , round(self.max_deck_power/self.min_deck_size*16/3/3)+1 ) # stdev param for normal distribution when generating random decks

        # player attributes
        self.player_decks = { 0 : Deck(self.min_deck_size,self.max_deck_power) , 1 : Deck(self.min_deck_size,self.max_deck_power) }
        self.num_unplayed_cards = { 0 : 0 , 1 : 0 } # initialize as the size of the corresponding deck and then decrement everytime the corresponding player plays a card
        self.player_hands = { 0 : [] , 1 : [] } # each item is a list[ 10 ints ], representing a card; 0 would represent an empty card
        self.player_points = { 0 : 0 , 1 : 0 } # total points (sum of card power) each player currently has for this round
        self.player_num_round_wins = { 0 : 0 , 1 : 0 } # keeps track of how many rounds each player won (this game is played as best 2 out of 3
        self.player_total_remaining_card_power = { 0 : 0 , 1 : 0 } # initialize at self.max_deck_power and then decrement by an amount equal to the card played by the corresponding player
        self.player_average_remaining_card_power = { 0 : 0 , 1 : 0 } # divide corresponding value in self.player_total_remaining_card_power by self.num_unplayed_cards
        
        # game attributes
        self.round = 0 # 1-indexed, 3 rounds
        self.active_players = [] # keep track of which player indices are still active in this round
        self.active_player_index = None # indexes self.active_players and indicates whose turn it is
        self.round_one_first_player_index = None # keep track of who started first in round 1; to be used when deciding who goes first if the round ties
        
    def get_name(self): return 'Gwent Lite'
    def get_observation_shape(self): return (self.max_card_power+10+5) + (7) + (2)
    def get_action_space_size(self): return 11 #77 # play any of the 10 cards in your hand (actions 1-10) or pass (action 0)
    def get_number_of_players(self): return 2
    def reset(self,deck_lists=(None,None)): # deck_lists is a list[ deck0_list[int]/None , deck0_list[int]/None ]
        for player_index in range(2):
            self.player_decks[player_index].reset(deck_lists[player_index],self.mean,np.random.choice(self.stdev_range))
            self.num_unplayed_cards[player_index] = len(self.player_decks[player_index].deck)
            self.player_hands[player_index].clear()
            for _ in range(10): self.player_hands[player_index].append( self.player_decks[player_index].draw() )
            self.player_points[player_index] = 0
            self.player_num_round_wins[player_index] = 0
            self.player_total_remaining_card_power[player_index] = self.max_deck_power
            self.player_average_remaining_card_power[player_index] = self.player_total_remaining_card_power[player_index] / self.num_unplayed_cards[player_index]

        self.round = 1
        self.active_players = [0,1]
        self.active_player_index = np.random.choice(2) # randomly pick a player to start first
        self.round_one_first_player_index = self.get_player_turn() # assign self.round_one_first_player_index to the player index that starts first in round 1
    def next_round(self):
        # increment self.round, reset self.active_players, increment the winner's score in self.player_num_round_wins (if tie, increment both), set self.active_player_index to the winner of the last round
        self.round += 1
        self.active_players = [0,1]

        if self.player_points[0] == self.player_points[1]: # tie (if this happens in round 2, game is over, as someone has 2 wins)
            for player_index in range(2): self.player_num_round_wins[player_index] += 1
            self.active_player_index = (self.round_one_first_player_index+1) % 2 # if tie, the player that didn't start first in round 1 will start first in the next round
        else: # one player wins, one player loses
            winner_index = np.argmax( ( self.player_points[0] , self.player_points[1] ) )
            self.player_num_round_wins[winner_index] += 1
            self.active_player_index = winner_index

        # reset self.player_points, and have each player draws 3 cards (or until they have 10 cards in hand)
        for player_index in range(2):
            self.player_points[player_index] = 0
            for _ in range( min( 10-len(self.player_hands[player_index]) , 3 ) ): self.player_hands[player_index].append( self.player_decks[player_index].draw() )
    def get_player_turn(self): return self.active_players[self.active_player_index] # return player index of active player
    def act(self,action):
        active_player_index = self.get_player_turn()

        if action == 0 or action-1 >= len(self.player_hands[ active_player_index ]): # pass if action==0 or action points to a blank card (0); i.e. it's impossible to make an illegal move because that would just default to pass
            self.active_players.remove( active_player_index )
            if len(self.active_players) == 0: self.next_round()
            else: self.active_player_index = (self.active_player_index+1) % len(self.active_players)
            return True

        if action-1 < len(self.player_hands[ active_player_index ]): # actions 1-10 correspond to playing a card in the player's hand
            card_played = self.player_hands[ active_player_index ].pop(action-1)
            
            self.player_points[ active_player_index ] += card_played # play the card from active_player_hand and increment that player's points by that card's power
            # update player attributes
            self.num_unplayed_cards[ active_player_index ] -= 1
            self.player_total_remaining_card_power[ active_player_index ] -= card_played
            self.player_average_remaining_card_power[ active_player_index ] = self.player_total_remaining_card_power[ active_player_index ] / self.num_unplayed_cards[ active_player_index ]

            if len(self.player_hands[ active_player_index ]) == 0: # if that was the player's last card
                self.active_players.remove( active_player_index ) # remove the player from self.active_players
                if len(self.active_players) == 0: self.next_round() # go to next round if there are no more active players
                else: self.active_player_index = (self.active_player_index+1) % len(self.active_players) # otherwise, increment self.active_player_index to the next player
            else: self.active_player_index = (self.active_player_index+1) % len(self.active_players)
            
            return True

    def check_game_over(self):
        players_with_2_round_wins = []
        for player_index in range(2):
            if self.player_num_round_wins[player_index] == 2: players_with_2_round_wins.append(player_index)

        if len(players_with_2_round_wins) == 0: return False,None
        if len(players_with_2_round_wins) == 1: return True , { players_with_2_round_wins[0] : 'win' , (players_with_2_round_wins[0]+1)%2 : 'loss' }

        # if we reach this point, that means len(players_with_2_round_wins) == 2 and it's a tie
        return True , {0:'tie',1:'tie'}
    def get_features(self,player_index):
        # Active player features: deck_onehot(self.max_card_power), hand(10), num_unplayed_cards(scalar), points(scalar), num_round_wins(scalar), total_remaining_power(scalar), average_remaining_power(scalar)
        # Opponent features: deck_size(scalar), hand_size(scalar), num_unplayed_cards(scalar), points(scalar), num_round_wins(scalar), total_remaining_power(scalar), average_remaining_power(scalar)
        # Game features: round_num(scalar), num_active_players(scalar)
        
        opponent_index = (player_index+1) % 2

        return np.concatenate( (
            
            # active player features
            self.player_decks[player_index].get_features(), # player's deck features (already normalized)
            [ self.player_hands[player_index][i]/self.max_card_power if i < len(self.player_hands[player_index]) else 0 for i in range(10) ], # player's hand features, normalized by dividing by the max card power that can exist given game settings
            [ self.num_unplayed_cards[player_index] / self.min_deck_size , # number of cards unplayed by player, normalized by dividng by the minimum deck size
              self.player_points[player_index] / ( self.max_card_power + 9 ), # player points, normalized by dividing by the max power a player could possibly play in a single round
              self.player_num_round_wins[player_index] / 2, # number of player round wins, normalized by dividing by 2
              self.player_total_remaining_card_power[player_index] / self.max_deck_power, # total remaining card power of player, normalized by dividing by max total card power (100)
              self.player_average_remaining_card_power[player_index] / self.max_card_power ], # average remaining card power of player, normalized by dividing by max card power (76)

            # opponent features
            [ len(self.player_decks[opponent_index].deck) / self.min_deck_size, # opponent deck size, normalized by dividing by minimum deck size
              len(self.player_hands[opponent_index]) / 10, # opponent hand size, normalized by dividing by 10 (the max hand size)
              self.num_unplayed_cards[opponent_index] / self.min_deck_size , # number of cards unplayed by opponent, normalized by dividng by the minimum deck size
              self.player_points[opponent_index] / ( self.max_card_power + 9 ), # opponent points, normalized by dividing by the max power a player could possibly play in a single round
              self.player_num_round_wins[opponent_index] / 2, # number of opponent round wins, normalized by dividing by 2
              self.player_total_remaining_card_power[opponent_index] / self.max_deck_power, # total remaining card power of opponent, normalized by dividing by max total card power (100)
              self.player_average_remaining_card_power[opponent_index] / self.max_card_power ], # average remaining card power of opponent, normalized by dividing by max card power (76)

            # game features
            [ self.round / 3 , len(self.active_players) / 2 ]

            ) ).reshape(1,-1)

    def sample_legal_move(self):
        return np.random.choice( len( self.player_hands[ self.get_player_turn() ] ) + 1 ) # pass is action 0, while actions 1-10 correspond to playing a card in the player's hand
    def __str__(self): return f'\nPlayer decks: {self.player_decks}\nPlayer hands: {self.player_hands}\nPlayer points: {self.player_points}\nPlayer round wins: {self.player_num_round_wins}\nPlayer total remaining card power: {self.player_total_remaining_card_power}\nPlayer average remaining card power: {self.player_average_remaining_card_power}\nRound: {self.round}\nPlayer turn: {self.get_player_turn()}'
    def __repr__(self): return str(self)
    def play(self): # play a human vs human game
        while True:
            self.reset()

            game_over,winner = self.check_game_over()
            while not game_over:
                print(self)
                self.get_features(self.get_player_turn())
                
                a = int( input('Move: ') )
                while not self.act(a): a = int( input('Illegal!\nMove: ') )
                game_over,winner = self.check_game_over()
            print(self)
            self.get_features(self.get_player_turn())

            print(self.check_game_over())

if __name__ == '__main__':
    g = GwentLite()
    g.play()
