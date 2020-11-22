# Reinforcement Learning Repo
This is a repo where I test reinforcement learning algorithms on board games. Details on how the board games and algorithms are implemented can be found below.
## Games list
* [Tic Tac Toe](#tic-tac-toe)
* Connect Four
* Incan Gold

### Tic Tac Toe
* A 2 player perfect information game
* Observations of the game state were represented using two 3 x 3 binary matrices (one for each player)
    * the binary matrices contain a value of 1 if a piece corresponding to that player is occupying that corresponding space on the 3 x 3 grid, 0 otherwise
* the binary matrices are combined together to make a 2 x 3 x 3 binary matrix feature representation of the game state
    * the first slice always contains the current active player's pieces
    * the second slice always contains the opponent's pieces (in the perspective of the current active player)