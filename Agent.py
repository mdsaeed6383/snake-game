"""
CSC 580 HW#4 "Agent.py" -- Class Agent, which performs Temporal Difference (TD) Q-Learning.

"""
import random
import numpy as np
import csv

class Agent:
    """ 
    An AI agent which controls the snake's movements.
    """
    def __init__(self, env, params):
        self.env = env
        self.action_space = env.action_space  # 4 actions for SnakeGame
        self.state_space = env.state_space    # 12 features for SnakeGame
        self.gamma = params['gamma']
        self.alpha = params['alpha']
        self.epsilon = params['epsilon'] 
        self.epsilon_min = params['epsilon_min'] 
        self.epsilon_decay = params['epsilon_decay']
        ## TO-DO: Choose your data structure to hold the Q table and initialize it
        self.Q = {}
        

    @staticmethod
    def state_to_int(state_list):
        """ Map state as a list of binary digits, e.g. [0,1,0,0,1,1,1] to an integer."""
        return int("".join(str(x) for x in state_list), 2)
    
    @staticmethod
    def state_to_str(state_list):
        """ Map state as a list of binary digits, e.g. [0,1,0,0,1,1,1], to a string e.g. '0100111'. """
        return "".join(str(x) for x in state_list)

    @staticmethod
    def binstr_to_int(state_str):
        """ Map a state binary string, e.g. '0100111', to an integer."""
        return int(state_str, 2)

    # (A) 
    def init_state(self, state):
        """ Initialize the state's entry in state_table and Q, if anything needed at all."""
        state_key = self.state_to_int(state)
        if state_key not in self.Q:
            self.Q[state_key] = np.zeros(self.action_space)  # Initialize Q-values to zero
        
        
    # (A)
    def select_action(self, state):
        """
        Do the epsilon-greedy action selection. Note: 'state' is an original list of binary digits.
        It should call the function select_greedy() for the greedy case.
        """
        state_key = self.state_to_int(state)
        self.init_state(state)  # Ensure state is in Q-table

        if random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_space)  # Explore
        else:
            return np.argmax(self.Q[state_key])  # Exploit (greedy action)

    # (A)
    def select_greedy(self, state):
        """ 
        Greedy choice of action based on the Q-table. 
        """
        state_key = self.state_to_int(state)
        self.init_state(state)  # Ensure state is in Q-table
        return np.argmax(self.Q[state_key])
    
    # (A)
    def update_Qtable(self, state, action, reward, next_state):
        """
        Update the Q-table (and anything else necessary) after an action is taken.
        Note that both 'state' and 'next_state' are an original list of binary digits.
        """
        state_key = self.state_to_int(state)
        next_state_key = self.state_to_int(next_state)

        self.init_state(state)
        self.init_state(next_state)

        # Compute Q-learning update
        best_next_action = np.max(self.Q[next_state_key])  # max_a Q(S', a)
        self.Q[state_key][action] += self.alpha * (reward + self.gamma * best_next_action - self.Q[state_key][action])

        # Decay epsilon to reduce exploration over time
        self.adjust_epsilon()
        

    # (A)
    def num_states_visited(self):
        """ Returns the number of unique states visited. Obtain from the Q table."""
        return len(self.Q)
    
    # (A)
    def write_qtable(self, filepath):
        """ Write the content of the Q-table to an output file. """
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["State", "Action", "Q-Value"])
            for state, actions in self.Q.items():
                for action, q_value in enumerate(actions):
                    writer.writerow([state, action, q_value])
    # (A)
    def read_qtable(self, filepath):
        """ Read in the Q table saved in a csv file. """
        self.Q = {}
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row
            for row in reader:
                state, action, q_value = int(row[0]), int(row[1]), float(row[2])
                if state not in self.Q:
                    self.Q[state] = np.zeros(self.action_space)
                self.Q[state][action] = q_value


    def adjust_epsilon(self):
        """ Implements the epsilon decay. """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
