import gymnasium as gym
import numpy as np
import random

# Creating a Taxi Environment
env = gym.make('Taxi-v3')

# The agent takes a random action on each time step for a total of 200 time steps. we are running a total of 1500 episodes and each episode has 200 time steps.

agent_rewards_per_episode = []
state = env.reset()

for episode in range(1500):
    agent_reward = 0
    
    for s in range(201):
        # getting a random action
        action = env.action_space.sample()

        # perform the action on the environment
        (new_state,reward, ter, tru, info) = env.step(action)
        
        # update the agent's reward 
        agent_reward = agent_reward + reward

        env.render()
    
    # storing agent's reward for each episode 
    agent_rewards_per_episode.append(agent_reward)

# close the environment
env.close()


# Plotting code
import matplotlib.pyplot as plt
import numpy as np

# Define the x and y data
iterations = [i for i in range(1,1501,100)]

# averaging the rewards for every 100 episodes out of 1500. so that we can get smooth graph.
average_qagent_rewards = []
for i in range(0, len(agent_rewards_per_episode), 100):
    slice = agent_rewards_per_episode[i:i+100]
    mean = sum(slice) / len(slice)
    average_qagent_rewards.append(mean)

# Create the plot
plt.plot(iterations, average_qagent_rewards,color='green')

# Add labels and title
plt.xlabel('Num of iterations')
plt.ylabel('Reward per iteration')
plt.title('Untrained agent\'s performance')

# Show the plot
plt.show()