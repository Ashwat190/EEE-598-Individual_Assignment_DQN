# Creating RelayBuffer file for DQN, managing q values

import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=10000):  
        # capacity = maximum number of experiences the buffer can store
        self.capacity = capacity
        
        # these lists will store each component of experience separately
        self.states = []        
        self.actions = []       
        self.rewards = []       
        self.next_states = []   
        self.dones = []         

    def add(self, state, action, reward, next_state, done): # Creating the SARS function for updating the q values

        # Append the new data to respective lists
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

        # If buffer exceeds its capacity, remove the oldest experience (FIFO)
        if len(self.states) > self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)

    def sample(self, batch_size=64): # Function for testing the batches

        # Choose random indices for sampling
        indices = np.random.choice(len(self.states), size=batch_size, replace=False)
        
        # Gather corresponding samples
        batch_states = np.array([self.states[i] for i in indices])
        batch_actions = np.array([self.actions[i] for i in indices])
        batch_rewards = np.array([self.rewards[i] for i in indices])
        batch_next_states = np.array([self.next_states[i] for i in indices])
        batch_dones = np.array([self.dones[i] for i in indices])

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones

    def __len__(self): # Returns the len of the states
        return len(self.states)
