# For testing the trained DQN using the trained weights

from GridWorld import GridWorld
from Q_NN import Qnetwork
import torch
import numpy as np
import matplotlib.pyplot as plt 

# Load trained model
model = Qnetwork()
model.load_state_dict(torch.load("qnetwork.pth"))  # Calls the trained weights for evaluation
model.eval()  # set model to evaluation mode

# Set up evaluation environment
env = GridWorld()
n_episodes = 200

total_rewards = []
success_count = 0
steps_list = []

for ep in range(n_episodes):
    state = env.reset()
    done = False
    ep_reward = 0
    steps = 0

    while not done:
        # Convert state
        idx = state[0] * 4 + state[1]
        state_tensor = torch.zeros(12)
        state_tensor[idx] = 1.0

        # Greedy action without epsilon
        with torch.no_grad():
            q_values = model(state_tensor)
            action_idx = torch.argmax(q_values).item()
            action = env.get_actions()[action_idx]

        # For every step in the environment
        next_state, reward, done = env.step(action)

        ep_reward += reward
        steps += 1
        state = next_state

        if done and state == (0,3): # For counting the success states
            success_count += 1 

    total_rewards.append(ep_reward)
    steps_list.append(steps)

# Calculating the average stats for evaluation
avg_reward = np.mean(total_rewards)
avg_steps = np.mean(steps_list)
success_rate = success_count / n_episodes * 100

print("\n--- Evaluation Results ---") # Prints the important results of evaluation
print(f"Episodes = {n_episodes}")
print(f"Average Reward = {avg_reward:.2f}")
print(f"Average Steps per Episode = {avg_steps:.2f}")
print(f"Success Rate (reached +1) = {success_rate:.1f}%")
print(f"Max Reward = {round(max(total_rewards),2)}")

# Printing the optimal policy 
actions = ["N", "S", "W", "E"]
policy_grid = []

for r in range(3):
    row = []
    for c in range(4):
        state = (r, c)
        
        # Skip wall
        if state == env.wall:
            row.append("X")
            continue
        
        # Terminal states
        if state in env.terminal_states:
            reward = env.terminal_states[state]
            row.append(f"{reward:+.0f}")
            continue

        # Evaluate greedy action for this state
        idx = r * 4 + c
        state_tensor = torch.zeros(12)
        state_tensor[idx] = 1.0
        with torch.no_grad():
            q_values = model(state_tensor)
            best_action = actions[torch.argmax(q_values).item()]
        row.append(best_action)
    policy_grid.append(row)

print("\nOptimal Policy")
for row in policy_grid:
    print(row)

# Plotting the evaluation outcome 
plt.figure(figsize=(10, 5))
plt.plot(total_rewards, label="Reward per Episode", color='blue')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Evaluation Results")
plt.legend()
plt.grid(True)
plt.show()

