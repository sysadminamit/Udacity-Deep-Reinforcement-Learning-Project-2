# Project Report 
## Udacity Deep Reinforcement Learning Project 2: Continuous Control 

The purpose of this document is to describe the details of the project; including the algorithm, network architecture, training hyperparameters and results.

In this report I briefly summarize the learnings and final modeling decisions taken as part of the Continuous Control project.I was able to find a setup that solves the environment with around 250 steps but it took some optimizations from the original DDPG code from the Udacity repository to make it work for me.

## Project Overview

[//]: # (Image References)

[image1]: https://github.com/joshnewnham/Udacity_DeepReinforcementLearning_Project2/blob/master/images/reacher.gif "Agent"

![Agent][image1]

The goal of this project was to train an agent to control a double-jointed arm such that it would track a ball around. As described in the project: A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## Learning Algorithm

In this section we briefly describe the learning algorithm, along with the network architectures used in this project.

Reinforcement learning algorithms can be categoized as either value-based, policy-based or combination of the two. Value-based learning consists of learning a state-action value function (Value/Q-function) that leads to the highest-valued state, by contract the policy-based approach tries to directly learn a (optimal) policy function directly (without the intermediary Value/Q-function).

In the previous [project-1](https://github.com/sysadminamit/Udacity-Deep-Reinforcement-learning-Project-1) we used a value-based algorithm, Deep Q-Network (DQN), to successfuly train an agent to navigate an environment scattered with good and bad bananas. DQN has seen much success dealing with environments with high-dimensional (complex) states but only dealing with discrete actions. Unfortunately value-based algorithms don't scale well when the action space is large, such as when they require a continous output (as its very difficult to converge for large action spaces) such as what is required in this project.

Deep Deterministic Policy Gradient (DDPG) (the algorithm used in this project) builds on DPG (mentioned above) but introduces an actor-critic architecture to deal with a large action space (continous or discrete).

An actor is used to tune the parameters 𝜽 for the policy function i.e. decide the best action for a specific state while a critic is used to evaluate the policy function estimated by the actor according to the temporal difference (TD) error (TD learning is a way to learn how to predict a value depending on future values for a given state, similar to Q-Learning).

The figure below illustrates the Actor-critic Architecture (source [Continous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf)).

![ ](Images/actor_critic_architecture_image.png)


## Network Architectures.
Throughout my experimentation phase I did not change the basic network architecture much in terms of number of layers and number of units: it was always 2 fully connected hidden layers with ReLu activations for both the actor and the critic. I tried 64/64, 128/64 and 128/128 units for the two hidden layers and used 128/128 in my final setup. The main improvement to get the agent to train came from a suggestion in the Nanodegree Slack channel: When introducing batch normalization after the first hidden layer for the actor network the training started to get somewhere. Before (with the standard feedforward network and some optimizations outlined below), the training would stall at average score of 1 − 2. I later decided to add one batch normalization layer also in the critic network. Looking at the training progress (see chart below), the smoothing/regularizing effect of batch normalization was pronounced in the sense that training progress started visibly after only a handful of episodes.

