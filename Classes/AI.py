import copy
import random
import pickle
import os
from Classes.BoardClass import *


class AI:
    def __init__(self, level, player, Q=None):
        """
        level=0 => random
        level=1 => minimax
        level=2 => Q-learning
        player: the player the AI controls (1 or 2)
        Q: shared Q-table (dictionary)
        """
        self.level = level
        self.player = player
        # Q is a dict keyed by ((state_key), (row, col)) -> float
        self.Q = Q if Q is not None else {}

    # -----------------------------
    #  EVALUATE: Called During Game
    # -----------------------------
    def eval(self, board):
        """
        Decide on a move based on self.level:
          0 => random
          1 => minimax
          2 => Q-learning (lookup best from Q-table)
        """
        if self.level == 0:
            chosen_move = self.rnd(board)
            chosen_method = "random"
        elif self.level == 1:
            # If self.player=1, we want the 'maximizing' perspective;
            # if self.player=2, we want the 'minimizing' perspective.
            maximizing = True if self.player == 1 else False
            score, move = self.minimax(board, maximizing)
            chosen_move = move
            chosen_method = "minimax"
        else:
            # Q-learning: we look up best action for self.player
            best_move = self.get_best_action_from_Q(board, self.player)
            if best_move is None:
                # fallback if unseen state
                best_move = self.rnd(board)
                chosen_method = "Q-random"
            else:
                chosen_method = "Q-learning"
            chosen_move = best_move

        print(f"[AI] Chosen move {chosen_move} with method: {chosen_method}")
        return chosen_move

    # -----------------------------
    #  LEVEL 0: Random Move
    # -----------------------------
    def rnd(self, board):
        """returns a random valid move"""
        empty_sqrs = board.get_empty_sqrs()
        return random.choice(empty_sqrs)

    # -----------------------------
    #  LEVEL 1: Minimax Algorithm
    # -----------------------------
    def minimax(self, board, maximizing):
        """
        Simple minimax to handle 3x3 Tic-Tac-Toe.
        'maximizing' indicates if we are maximizing X (player=1).
        If your self.player =2, that becomes the minimizing side.
        """
        case = board.final_state()
        # Check terminal states
        if case == 1:  # X wins
            return (100, None)
        if case == 2:  # O wins
            return (-100, None)
        if board.isfull():
            return (0, None)

        if maximizing:
            best_eval = float('-inf')
            best_move = None
            for (r, c) in board.get_empty_sqrs():
                temp = copy.deepcopy(board)
                temp.mark_sqr(r, c, 1)  # X
                eval_score, _ = self.minimax(temp, False)
                if eval_score > best_eval:
                    best_eval = eval_score
                    best_move = (r, c)
            return (best_eval, best_move)
        else:
            best_eval = float('inf')
            best_move = None
            for (r, c) in board.get_empty_sqrs():
                temp = copy.deepcopy(board)
                temp.mark_sqr(r, c, 2)  # O
                eval_score, _ = self.minimax(temp, True)
                if eval_score < best_eval:
                    best_eval = eval_score
                    best_move = (r, c)
            return (best_eval, best_move)

    # -----------------------------
    #  LEVEL 2: Q-Learning
    # -----------------------------
    def make_state_key(self, board, current_player):
        """
        Encode the board + current player as a unique string.
        """
        squares_str = ''.join(str(int(x)) for x in board.squares.flatten())
        return f"{squares_str}_p{current_player}"

    def get_best_action_from_Q(self, board, current_player):
        """
        Look up the best action for current_player from the Q-table.
        """
        state_key = self.make_state_key(board, current_player)
        valid_moves = board.get_empty_sqrs()
        best_move = None
        best_q = float('-inf')
        for move in valid_moves:
            q = self.Q.get((state_key, move), 0.0)
            if q > best_q:
                best_q = q
                best_move = move
        return best_move

    # -----------------------------
    #  SELF-PLAY TRAINING METHOD
    # -----------------------------

    def self_play_training(self, episodes, alpha, gamma, epsilon):
        """
        Train two Q-learning agents (one for each player) by letting them play against each other.
        Each agent records its own history; at the end of the game, each agent's history is used
        to update the shared Q-table using a discounted Monte Carlo update.
        """
        # Shared Q-table
        Q = {}
        # Create two agents: one for player 1 and one for player 2.
        agent1 = AI(level=2, player=1, Q=Q)
        agent2 = AI(level=2, player=2, Q=Q)


        for ep in range(episodes):

            board = Board()
            done = False
            # Separate histories for each agent
            #history_agent1 = []
            history_agent2 = []
            # Let player 1 (agent1) start the game.
            current_agent = agent1

            while not done:
                state_key = current_agent.make_state_key(board, current_agent.player)
                # Epsilon-greedy action selection.
                if random.random() < epsilon:
                    action = current_agent.rnd(board)
                else:
                    action = current_agent.get_best_action_from_Q(board, current_agent.player)
                    if action is None:
                        action = current_agent.rnd(board)

                # Save the move in the appropriate history.
                if current_agent.player == 1:
                    history_agent1.append((state_key, action))
                else:
                    history_agent2.append((state_key, action))

                # Execute the move.
                board.mark_sqr(action[0], action[1], current_agent.player)

                # Check for terminal condition.
                if board.final_state() != 0 or board.isfull():
                    done = True
                else:
                    # Switch agent for the next move.
                    current_agent = agent2 if current_agent == agent1 else agent1

            # Determine game outcome rewards.
            final = board.final_state()  # 1 if player 1 wins, 2 if player 2 wins, 0 for draw.
            if final == 1:
                reward1, reward2 = 1, -1
            elif final == 2:
                reward1, reward2 = -1, 1
            else:
                reward1, reward2 = 0, 0

            # Update Q-table for agent1's moves.
            G = 0
            for state_key, action in reversed(history_agent1):
                G = reward1 + gamma * G
                old_q = Q.get((state_key, action), 0.0)
                Q[(state_key, action)] = old_q + alpha * (G - old_q)

            # Update Q-table for agent2's moves.
            G = 0
            for state_key, action in reversed(history_agent2):
                G = reward2 + gamma * G
                old_q = Q.get((state_key, action), 0.0)
                Q[(state_key, action)] = old_q + alpha * (G - old_q)

        self.Q = Q  # Update this agent's Q-table with the shared one.
        return Q

    # -----------------------------
    #  SAVE / LOAD Q
    # -----------------------------
    def save_Q(self, filename='qtable.pkl'):
        with open(filename, 'wb') as f:
            print("Q-table loaded. Size:", len(self.Q))
            pickle.dump(self.Q, f)

    def load_Q(self, filename='qtable.pkl'):
        print("[AI] Loading Q-table from:", filename)
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.Q = pickle.load(f)
            print("Q-table loaded. Size:", len(self.Q))
        else:
            print(f"File {filename} not found!")
