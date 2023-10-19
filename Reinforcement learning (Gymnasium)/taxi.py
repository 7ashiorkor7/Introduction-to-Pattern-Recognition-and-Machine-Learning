""" OpenAI Gym – Taxi (V3) Environment 
In this exercise we will use the Gym environment (https://www.gymlibrary.dev/). You may install it
via Anaconda (see the course Moodle page). Launch Python and type the following commands:
$> python
>>> import gym
>>> env = gym.make("Taxi-v3",render_mode=’ansi’)
>>> env.reset()
>>> print(env.render())
You should see a map with four locations. Read the description of the map from the source: https:
//www.gymlibrary.dev/environments/toy_text/taxi/ Use the commands 0-5 and solve the problem
manually (render after each step):
>>> state, reward, done, truncated, info = env.step(1)
>>> print(env.render())
Your task is to implement Q-learning to solve the Taxi problem with optimal policy. For this you need to
fill in the missing parts in the openai taxi skeleton.py available in the course Moodle page. By default the
skeleton creates a random Q-table, you may try:
$> python openai_taxi_skeleton.py
When you have implemented the training part, then run your method ten times and compute the average
total reward and the average number of actions. """



# Load OpenAI Gym and other necessary packages
import gym
import random
import numpy
import time

# Environment
env = gym.make("Taxi-v3", render_mode='ansi')

# Training parameters for Q learning
alpha = 0.9 # Learning rate
gamma = 0.9 # Future reward discount factor
num_of_episodes = 1000
num_of_steps = 500 # per each episode

# Q tables for rewards
#Q_reward = -100000*numpy.ones((500,6)) # All same
Q_reward = -100000*numpy.random.random((500, 6)) # Random

# Training w/ random sampling of actions
def train(env, Q_reward, alpha, gamma, num_of_episodes, num_of_steps):
  for episode in range(num_of_episodes):
    state = env.reset()[0]
    total_reward = 0
    for step in range(num_of_steps):
      action = numpy.random.choice(6)
      next_state, reward, done, truncated, info = env.step(action)
      total_reward += reward
      # Update Q-table
      max_Q_next_state = numpy.max(Q_reward[next_state,:])
      Q_reward[state, action] = Q_reward[state, action] + alpha * (reward + gamma * max_Q_next_state - Q_reward[state, action])
      state = next_state
      if done:
        break
  return Q_reward

# Testing
def test(env, Q_reward):
  state = env.reset()[0]
  tot_reward = 0
  for t in range(50):
    action = numpy.argmax(Q_reward[state,:])
    state, reward, done, truncated, info = env.step(action)
    tot_reward += reward
    print(env.render())
    time.sleep(1)
    if done:
        print("Total reward %d" %tot_reward)
        break

# Train Q-table
Q_reward = train(env, Q_reward, alpha, gamma, num_of_episodes, num_of_steps)

# Test optimal policy
test(env, Q_reward)

# Compute average total reward and average number of actions over 10 runs
total_reward_sum = 0
total_actions_sum = 0
for i in range(10):
  state = env.reset()[0]
  tot_reward = 0
  num_actions = 0
  for t in range(50):
    action = numpy.argmax(Q_reward[state,:])
    state, reward, done, truncated, info = env.step(action)
    tot_reward += reward
    num_actions += 1
    if done:
        break
  total_reward_sum += tot_reward
  total_actions_sum += num_actions

avg_total_reward = total_reward_sum / 10
avg_num_actions = total_actions_sum / 10

# Print average total reward and average number of actions
print("Average total reward:", avg_total_reward)
print("Average number of actions:", avg_num_actions)
