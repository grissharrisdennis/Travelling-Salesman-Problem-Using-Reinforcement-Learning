# Travelling-Salesman-Problem-Using-Reinforcement-Learning
Created a reinforcement agent  to solve the Travelling Salesman Problem and used pygame to visualize the training.

This project implements a Q-learning agent to solve the classic Traveling Salesman Problem (TSP). The agent is trained to find the shortest route that visits a set of cities exactly once and returns to the starting point, optimizing the total distance traveled. The project uses the Pygame framework to visualize the agent’s learning process in real-time.

## Introduction
The Traveling Salesman Problem (TSP) is an optimization problem where the objective is to find the shortest possible route that visits a set of cities exactly once and returns to the starting city. This project leverages reinforcement learning, specifically the Q-learning algorithm, to address this problem by training an agent to learn the optimal path.

## Technologies Used
### Programming Language : Python
### Visualization Framework: Pygame
### Machine Learning Concept: Reinforcement Learning (Q-learning)

## How It Works
Q-learning Agent: The agent is trained using the Q-learning algorithm to find the optimal route.
Reward System: The reward is set as the negative distance between cities, encouraging the agent to minimize the total travel distance.
Installation
Visualization: Pygame is used to render the cities, paths, and the learning process, allowing you to see how the agent improves over time.

To run this project locally, follow these steps:
### Clone the repository:
git clone https://github.com/grissharrisdennis/Travelling-Salesman-Problem-Using-Reinforcement-Learning.git

cd Travelling-Salesman-Problem-Using-Reinforcement-Learning

### Install the required dependencies:
pip install -r requirements.txt
### Run the project:
python agent.py

## Usage
Run the project to see the agent's progress in learning the shortest path.
Modify the number of cities, learning rate, and other parameters in the main.py file to experiment with different setups.

