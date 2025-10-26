# Creating Neural Network to find Qvalues

import torch
import torch.nn as nn 

class Qnetwork(nn.Module): # Inheriting functions from nn (Module/file) which has class named Module
    def __init__(self, input_dim = 12, hidden_dim = 64, output_dim = 4 ): # Setting the  hyperparameters of the input, output and hidden layers.
        super(Qnetwork, self).__init__() # Calling constructor of parent class

        self.net = nn.Sequential( # Organizing neural network in sequential order. 
            nn.Linear(input_dim, hidden_dim), # First layer from input to neurons
            nn.ReLU(), # Activation function applied for the first layer
            nn.Linear(hidden_dim, hidden_dim), # Second layer from one hidden layer to another
            nn.ReLU(), # Activation function ReLu again
            nn.Linear(hidden_dim, output_dim) # Final layer from hidden layer to output (Q values) 
        )

    def forward(self, x): 
        return self.net(x)
