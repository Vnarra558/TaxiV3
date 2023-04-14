#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Venkat Narra and Sai Siddhardha Maguluri
# Created Date: 01/28/2023
# --------------------------------------------------------------------------

# Credit:
# using the enviroment from https://gymnasium.farama.org/environments/toy_text/taxi/. all the state, action, reward data will be provided by the environment itself.
# Q-Learning algorithm is taken from the textbook Reinforcement Learning(Richard S. Sutton and Andrew G. Barto), page no: 131
# SARSA Learning algorithm is taken from same textbook, page no: 130


import numpy as np
import gymnasium as gym

def trainAgentWithQLearning():

    # creating a Taxi environment with rgb_array mode.
    # with human mode we can see the actual taxi environment graphics, but it is taking so much time for training.
    env = gym.make('Taxi-v3', render_mode="rgb_array")

    # initializing q-table
    states = env.observation_space.n
    actions = env.action_space.n
    qtable = np.zeros((states, actions))

    # hyperparameters
    learning_rate = 0.6 
    discount_rate = 0.8
    epsilon = 1.0
    decay_rate= 0.005

    # training variables
    num_episodes = 1500
    max_steps = 200 # per episode

    # training
    for episode in range(1,num_episodes+1):

        # reset the environment
        state = env.reset()
        state = state[0]     
        iteration_reward = 0   

        for s in range(max_steps):

            # epsilon greedy policy
            if np.random.random() < epsilon:
                # explore
                action = env.action_space.sample()
            else:
                # exploit
                action = np.argmax(qtable[state,:])

            (new_state, reward, ter, tru, info)= env.step(action)
            
            iteration_reward = iteration_reward + reward

            # Q-learning algorithm
            qtable[state,action] = qtable[state,action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:])-(qtable[state,action]))

            # Update to our new state
            state = new_state
        

        # After every iteration, writing the reward that we get to a file 
        f = open("QLearning_Training_rewards.txt", "a")
        f.write("Iteration {} completed and the Q-Learning agent's reward is {} \n".format(episode,iteration_reward))
        f.close()

        # Inverse decay of epsilon
        epsilon = np.exp(-decay_rate*episode)

        print("Iteration {} completed and the Q-Learning agent's reward is {}".format(episode,iteration_reward))

    
    # Writing the qtable content to a file for future references
    np.savetxt("QLearning_qtable.txt", qtable, fmt="%d")

    print(f"Training completed over {num_episodes} episodes")

    input("Press Enter to watch the trained Q-Learning agent in action")

    # Trained agent
    env.render()
    state = env.reset()
    state = state[0]
    agent_reward = 0

    for s in range(max_steps):
        action = np.argmax(qtable[state,:])
        (new_state, reward, ter, tru, info)= env.step(action)
        agent_reward += reward
        print("Time step {} and Q-Learning agent's current reward is: {}".format(s+1,agent_reward))
        env.render()
        state = new_state

    env.close()

trainAgentWithQLearning()