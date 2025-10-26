# Creating the main training logic behind DQN

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

from GridWorld import GridWorld
from Q_NN import Qnetwork               
from replaybuffer import ReplayBuffer

# Setting up the training parameters 
num_episodes = 3000       # total training episodes
max_steps = 50               # max steps allowed per episode
batch_size = 32              # minibatch size for training
gamma = 0.99                 # discount factor
epsilon = 1.0                # initial epsilon value
epsilon_min = 0.05           # epsilon threshold value
epsilon_decay = 0.999        # how much to reduce epsilon each episode
learning_rate = 0.001        # learning rate
target_update_freq = 10      

# Calling the env and qnet setup 
env = GridWorld()                                
q_net = Qnetwork()                               
target_net = Qnetwork()                          
target_net.load_state_dict(q_net.state_dict())   
target_net.eval()                                
buffer = ReplayBuffer(capacity=5000)
optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Storing trackers for plotting/printing and averaging the values
rewards_per_episode = []     # Total reward in each episode
losses = []                  # Average loss per episode
steps_per_episode = []       # Track number of steps per episode

# Main training loop for calculating q value with epsilon decay, exploration and exploitation 
for episode in range(num_episodes):
    state = env.reset()         
    total_reward = 0             
    done = False
    episode_losses = []          
    steps = 0                    

    for step in range(max_steps):
        # Convert state
        state_idx = state[0] * 4 + state[1]        # convert (row, col) to index
        state_onehot = torch.zeros(12)
        state_onehot[state_idx] = 1.0           

        # Choose action using epsilon greedy method
        if random.random() < epsilon:
            action = random.choice(env.get_actions())  # random action, explore
        else:
            with torch.no_grad():
                q_values = q_net(state_onehot)          # predict Q-values
                action_idx = torch.argmax(q_values).item()
                action = env.get_actions()[action_idx]  # greedy action, exploit

        # Take step in the environment
        next_state, reward, done = env.step(action)
        total_reward += reward
        steps += 1  # increment step count

        # Convert next_state 
        next_idx = next_state[0] * 4 + next_state[1]
        next_onehot = torch.zeros(12)
        next_onehot[next_idx] = 1.0

        # Store experience in replay buffer
        buffer.add(state_onehot.numpy(), env.get_actions().index(action), reward, next_onehot.numpy(), done)

        state = next_state  # move to next state

        # Train only if enough samples are available
        if len(buffer) >= batch_size:
            # Sample a batch
            s_batch, a_batch, r_batch, s2_batch, d_batch = buffer.sample(batch_size)

            # Convert to tensors
            states = torch.tensor(s_batch, dtype=torch.float32)
            actions = torch.tensor(a_batch, dtype=torch.int64)
            rewards = torch.tensor(r_batch, dtype=torch.float32)
            next_states = torch.tensor(s2_batch, dtype=torch.float32)
            dones = torch.tensor(d_batch, dtype=torch.float32)

            # Compute target Q-values using target network
            with torch.no_grad():
                q_next = target_net(next_states)
                q_next_max = torch.max(q_next, dim=1)[0]
                targets = rewards + gamma * (1 - dones) * q_next_max

            # Compute current Q-values from main network
            q_vals = q_net(states)
            q_selected = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Compute loss and backpropagate 
            loss = loss_fn(q_selected, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            episode_losses.append(loss.item())

        if done:
            break

    # Record total reward, loss, and steps for this episode
    rewards_per_episode.append(total_reward)
    losses.append(np.mean(episode_losses) if episode_losses else 0)
    steps_per_episode.append(steps)

    # Decay epsilon 
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if episode % target_update_freq == 0:
        target_net.load_state_dict(q_net.state_dict())

    # Printing information and average of number of steps of every 10 episode
    if (episode + 1) % 10 == 0:
        avg_recent_steps = np.mean(steps_per_episode[-10:]) 
        print(f"Episode {episode+1}/{num_episodes} | Reward: {total_reward:.2f} | "
              f"Epsilon: {epsilon:.3f} | Avg Loss: {losses[-1]:.4f} | "
              f"Avg Steps (last 10): {avg_recent_steps:.2f}")

# Saving the trained weights for evaluation 
torch.save(q_net.state_dict(), "qnetwork.pth")
print("\nTraining complete. Model saved as qnetwork.pth")

# Plotting the training outcome 
plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode, label="Total Reward per Episode", color='b')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Results")
plt.legend()
plt.grid()
plt.show()

