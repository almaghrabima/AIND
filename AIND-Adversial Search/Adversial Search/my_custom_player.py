
from sample_players import DataPlayer
import math
import random

_WIDTH = 11
_HEIGHT = 9
_SIZE = (_WIDTH + 2) * _HEIGHT - 2

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is auto matically managed for you)
        #if self.context is None:
        #    self.context = dict()
        depth = 1
        while True:
            action = self.alpha_beta_search(state, depth)
            self.queue.put(action)
            # used to calculate the maximum depth   
            #self.context['depth'] = depth 
            depth += 1
        #self.queue.put(random.choice(state.actions()))

        
    def alpha_beta_search(self, state, depth):
        """ Return the move along a branch of the game tree that
        has the best possible value.  A move is a pair of coordinates
        in (column, row) order corresponding to a legal move for
        the searching player.
        You can ignore the special case of calling this function
        from a terminal state.
        """
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        v = None
        for a in state.actions():
            v = self._min_value(state.result(a), alpha, beta, depth - 1)
            alpha = max(alpha, v)
            if v > best_score:
                best_score = v
                best_move = a
        # player lost
        if best_move == None:
            best_move = random.choice(state.actions())
        return best_move     

    def _min_value(self, state, alpha, beta, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if state.terminal_test():
            return state.utility(self.player_id)

        if depth <= 0:
            return self.heuristic(state)

        v = float("inf")
        for a in state.actions():
            v = min(v, self._max_value(state.result(a), alpha, beta, depth - 1))
            if v <= alpha: return v
            beta = min(beta, v)
        return v
    
    def _max_value(self, state, alpha, beta, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if state.terminal_test():
            return state.utility(self.player_id)

        if depth <= 0:
            return self.heuristic(state)

        v = float("-inf")
        for a in state.actions():
            v = max(v, self._min_value(state.result(a), alpha, beta, depth - 1))
            if v >= beta: return v
            alpha = max(alpha, v)
        return v
    
    def heuristic(self, state):
        """
        Combines all the heuristics together
        """
        return self.baseline_heuristic(state) + 1.5 * self.distance_heuristics(state)
    
    def baseline_heuristic(self, state):
        """
        baseline heuristic mentioned in the lecture
        """
        return (len(state.liberties(state.locs[self.player_id])) - len(state.liberties(state.locs[1-self.player_id])) )

    def distance_heuristics(self, state):
        """
        Combines the center and boundaries heuristics
        """
        col1 = (state.locs[self.player_id]) % (_WIDTH + 2)
        row1 = (state.locs[self.player_id]) // (_WIDTH + 2)
        if state.locs[1- self.player_id]:
            col2 = (state.locs[1- self.player_id]) % (_WIDTH + 2)
            row2 = (state.locs[1- self.player_id]) // (_WIDTH + 2)
        else:
            col2 = _WIDTH//2
            row2 = _HEIGHT//2
        return self.center_heuristic(row1, col1, row2, col2) + 1.5 * self.boundaries_heuristic(row1, col1, row2, col2) 
    
    def center_heuristic(self, row1, col1, row2, col2):
        """
         Center heuristic measures how far the player’s location is from the center
        """
        my_dist = abs(col1  - _WIDTH//2) + abs(row1 - _HEIGHT//2)
        his_dist = abs(col2  - _WIDTH//2) + abs(row2 - _HEIGHT//2)
        return (his_dist - my_dist)

    def boundaries_heuristic(self, row1, col1, row2, col2):
        """
        Boundaries heuristic measures how far the player’s location is from the 4 walls and takes the minimum one
        """
        my_dist = min(_WIDTH - col1, col1, row1, _HEIGHT - row1)
        his_dist = min(_WIDTH - col2, col2, row2, _HEIGHT - row2)
        return (my_dist - his_dist)
    
