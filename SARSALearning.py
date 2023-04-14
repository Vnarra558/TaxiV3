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

def trainAgentWithSARSALearning():

    # create a Taxi environment with human mode i.e; we can see the actual taxi env graphics
    env = gym.make('Taxi-v3', render_mode="rgb_array")

    # initializing q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    # hyperparameters
    learning_rate = 0.6 
    discount_rate = 0.8
    epsilon = 1.0
    decay_rate= 0.005

    # training variables
    num_episodes = 1500
    max_steps = 200 # per episode

    # training
    for episode in range(num_episodes):

        # reset the environment
        state = env.reset()
        state = state[0]     
        iteration_reward = 0  
         
        # epsilon greedy policy
        if np.random.random() < epsilon:
            # explore	
            action = env.action_space.sample()	
        else:	
            # exploit	
            action = np.argmax(qtable[state, :])

        for s in range(max_steps): 
            # take action and observe next state and reward	
            (new_state, reward, ter, tru, info)= env.step(action)

            # epsilon greedy policy	
            if np.random.random() < epsilon:	
                # explore	
                next_action = env.action_space.sample()	
            else:	
                # exploit	
                next_action = np.argmax(qtable[new_state, :])
            
            iteration_reward = iteration_reward + reward
            qtable[state, action] = qtable[state, action] + learning_rate * (reward + discount_rate * qtable[new_state, next_action] - qtable[state, action])	

            # Update to our new state
            state = new_state	
            action = next_action
        

        # After every iteration, writing the reward that we get to a file 
        f = open("SARSA_rewards.txt", "a")
        f.write("Iteration {} completed and the SARSA agent's reward is {} \n".format(episode,iteration_reward))
        f.close()

        # Decrease epsilon
        epsilon = np.exp(-decay_rate*episode)

        print("Iteration {} completed and the SARSA agent's reward is {}".format(episode,iteration_reward))

    
    # Writing the qtable content to a file for future references
    np.savetxt("Sarsa_qtable.txt", qtable, fmt="%d")

    print(f"Training completed over {num_episodes} episodes")

    input("Press Enter to watch the trained SARSA agent in action")

    # Trained agent
    env.close()
    env.render()
    state = env.reset()
    state = state[0]
    agent_reward = 0

    for s in range(max_steps):
        action = np.argmax(qtable[state,:])
        (new_state, reward, ter, tru, info)= env.step(action)
        agent_reward += reward
        print("Time step {} and SARSA agent's current reward is: {}".format(s+1,agent_reward))
        env.render()
        state = new_state

    env.close()

trainAgentWithSARSALearning()