# Creating 3x4 GridWorld for the agent

import numpy as np 

class GridWorld: #defining the gridworld class

    def __init__(self):
        self.n_rows = 3 
        self.n_columns = 4
        self.start = (2,0)
        self.wall = (1,1)

        self.terminal_states = {
            (0,3) : 1.0,
            (1,3) : -1.0
        }

        self.step_cost = -0.04
        self.p_intended = 0.8
        self.p_left = 0.1
        self.p_right = 0.1

        self.state = self.start
        self.actions = ["N","S","W","E"]
        
        self.rng = np.random.default_rng() # Numpy random number generator

    def reset(self): # To reset agent to starting state
        self.state = self.start
        return self.state
    
    def step(self, action): # Take a step based on the chosen action
        if self.state in self.terminal_states: #Check if agent is in any of the terminal states
            reward = self.terminal_states[self.state] # returns the reward or penalty based on the terminal state i.e returns either 1.0 or -1.0
            return self.state, reward, True
        
        actual_action = self.choose_stochastic_action(action) # Gets the action taken after stochasticity is also considered/done
        next_state = self.move(self.state, actual_action) # Gets the next state reached after taking the stochastic action


        if next_state in self.terminal_states: # If terminal state completes the episode with the obtained reward or penalty
            reward = self.terminal_states[next_state] 
            done = True
        else: # If not terminal state then adds the cost of step taken 
            reward = self.step_cost
            done = False

        self.state = next_state # Sets the state to the latest reached state
        return next_state, reward, done
    
    def choose_stochastic_action(self, action): # Defines the stochasticity of each taken action
        directions = ["N", "S", "W", "E"]
        left = {"N":"W", "S":"E", "E":"N", "W":"S"} # Dictionary for outcomes if agent slips to left
        right = {"N":"E", "S":"W", "E":"S", "W":"N"} # same for right

        options = [action, left[action], right[action]] # When action is given these are the possible places agent could move eg: [N,W,E]
        probs = [self.p_intended, self.p_left, self.p_right] # These are the probability of the above action happening eg: [0.8,0.1,0.1]

        return self.rng.choice(options, p=probs) # rng.choice is a inbuilt function which takes two params (first is the list of option, then p for probability which is created above)
    
    def move(self, state, action): # Find new position/state of agent after taking an action

        r, c = state # Unpacking the row and column from self.state. "state" is just a temporary self.state

        # Defining movement in the gridworld eg: action "N" leads means from (2,0) -> (1,0)
        if action == "N":
            r = r - 1
        elif action == "S":
            r = r + 1
        elif action == "W":
            c = c - 1
        elif action == "E":
            c = c + 1

        if r < 0 or r >= self.n_rows or c < 0 or c >=self.n_columns: #Checks out of bounds
            return state # Stays in same state if the actions leads to out of bounds
        
        if (r, c) == self.wall: # Checks if actions leads to hitting wall
            return state 
        
        return (r, c) # Valid move is returned if all above cases are passed.

    # Return available actions (for agent use)
    def get_actions(self):
        return self.actions

    # Return current agent state
    def get_state(self):
        return self.state
