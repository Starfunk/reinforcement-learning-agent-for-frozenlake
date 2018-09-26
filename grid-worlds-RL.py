#A Reinforcement Learning Agent capable of learning how to play OpenAI's
#FrozenLake environment and Taxi environment!

import gym
import gym_minigrid
import numpy as np
import random
import matplotlib.pyplot as plt
from os import system, name 
import time
import sys

env_choose = int(input("Press '1' to run FrozenLake-v0 \nPress '2' to run \
FrozenLakeNotSlippery-v0 \nPress '3' to run FrozenLake8x8-v0 \nPress\
 '4' to run Taxi-v2 \n"))

if env_choose == 1:
	print("------------------Running FrozenLake-v0------------------")
	env = gym.make('FrozenLake-v0')
elif env_choose == 2:
	print("------------------Running FrozenLakeNotSlippery-v0------------------")
	env = gym.make('FrozenLakeNotSlippery-v0')
elif env_choose == 3:
	print("------------------Running FrozenLake8x8-v0------------------")
	env = gym.make('FrozenLake8x8-v0')
else: 
	print("------------------Running Taxi-v2------------------")
	env = gym.make('Taxi-v2')

#Hyperparameters
episodes = int(input("Please enter in the number of episodes you would like to \
use to train the agent:\n"))
gamma = 0.9 #Determines discount for future rewards
epsilon = 0.0 #Determines exploration vs exploitation rate
learning_rate = 0.1 #Scales Q-value incrementation
punish_step = -0.04 #Step reward for moving in FrozenLake
punish_lose = -1 #Punishment for falling into a hole in FrozenLake

#For plotting progression of RL algorithm
success_x = []
all_scores = []

#Initialize Q-matrix: i.e. the policy (combined with argmax).
Q = np.zeros([env.observation_space.n, env.action_space.n])

for episode in range(episodes):
    #done only equals True when the goal has been achieved
    done = False
    #counter records number of steps, reward is reward from action,
    #G is cumulative reward
    G, reward, counter = 0,0,0
    state = env.reset()
    while done != True:	
        if random.uniform(0, 1) < epsilon:
            action =  env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        state2, reward, done, info = env.step(action)
        if reward == 0:
            reward = punish_step
        if done == True and reward < 1:
            reward = punish_lose
        Q_value = Q[state,action] 
        Q[state,action] += learning_rate * (reward + gamma * np.max(Q[state2])-Q[state,action])
        G += reward
        counter += 1
        state = state2   
    print('Episode {} Total Reward: {} counter: {}'.format(episode,G,counter))
    all_scores.append(G)
    success_x.append(episode)
 
#Define clear() which clears output in the terminal to allow us to animate
#FrozenLake and Taxi. Function taken from https://www.geeksforgeeks.org/clear-screen-python/
def clear(): 
    # for windows 
    if name == 'nt': 
        _ = system('cls') 
    # for mac and linux(here, os.name is 'posix') 
    else: 
        _ = system('clear') 
  
#Displaying an episode of the AI acting on the Q-matrix
done = False
G, reward, counter = 0,0,0
state = env.reset()
interval = 1 if env_choose == 4 else 0.2
count = 0	
while done != True:
    time.sleep(interval)
    action = np.argmax(Q[state])
    state2, reward, done, info = env.step(action) 
    Q[state,action] += learning_rate * (reward + gamma * np.max(Q[state2])-Q[state,action])
    G += reward
    counter += 1
    state = state2 
    env.render()
    sys.stdout.flush()
    clear()
    count += 1

print("---------------------Trained Q-Matrix----------------------")
print(Q)

#Display RL learning progression by plotting total scores of each episode
fig2 = plt.figure()
x = np.linspace(0,episodes, num=episodes, endpoint=False)
plt.scatter(x, all_scores,c='black')
plt.xlabel('Episode')
plt.ylabel('Total Score')
plt.show()
