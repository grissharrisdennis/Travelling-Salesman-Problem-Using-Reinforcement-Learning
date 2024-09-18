import pygame
import random
import math
import numpy as np
import time

# Initialize pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Traveling Salesman Problem")
background = pygame.image.load('map.png')
# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
background = pygame.transform.scale(background, (WIDTH, HEIGHT))
# City parameters
NUM_CITIES = 10
cities = [(random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50)) for _ in range(NUM_CITIES)]

city_image = pygame.image.load('buildings.png')
city_image = pygame.transform.scale(city_image, (30, 30))

def draw_background():
    screen.blit(background, (0, 0))

def draw_cities(cities):
    for city in cities:
        screen.blit(city_image, (city[0] - 15, city[1] - 15))

def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def draw_path(cities, path):
    for i in range(len(path) - 1):
        pygame.draw.line(screen, BLACK, cities[path[i]], cities[path[i + 1]], 2)
    pygame.draw.line(screen, BLACK, cities[path[-1]], cities[path[0]], 2)

class QLearningAgent:
    def __init__(self, num_cities, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.q_table = np.zeros((num_cities, num_cities))
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.num_cities = num_cities

    def choose_action(self, state, visited):
        available_cities = [i for i in range(self.num_cities) if i not in visited]
    
        if len(available_cities) == 0:
            return None

        if np.random.rand() < self.epsilon:
            return np.random.choice(available_cities)
        else:
            q_values = self.q_table[state, :]
            q_values = [(q, idx) for idx, q in enumerate(q_values) if idx not in visited]
            return max(q_values, key=lambda x: x[0])[1]

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state, :])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_delta = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_delta

# Initialize the agent
agent = QLearningAgent(NUM_CITIES)

# Training loop
for episode in range(1000):
    visited = set()
    current_city = np.random.randint(NUM_CITIES)
    total_reward = 0
    agent_path = [current_city]
    visited.add(current_city)

    while len(visited) < NUM_CITIES:
        action = agent.choose_action(current_city, visited)
        if action is None:
            break
        
        reward = -distance(cities[current_city], cities[action])
        next_city = action
        agent.update_q_table(current_city, next_city, reward, next_city)
        
        current_city = next_city
        visited.add(current_city)
        agent_path.append(current_city)
        
        # Draw the updated path and cities after each action
        draw_background()
        draw_cities(cities)
        draw_path(cities, agent_path)
        pygame.display.update()

        time.sleep(0.5)

    # Compute the total reward for the full tour
    total_reward = -sum(distance(cities[agent_path[i]], cities[agent_path[i + 1]]) for i in range(len(agent_path) - 1))
    total_reward -= distance(cities[agent_path[-1]], cities[agent_path[0]])  # Complete the loop

    print(f"Episode {episode} - Total reward: {total_reward}")

    # Refresh the screen after each episode
    draw_background()
    draw_cities(cities)
    draw_path(cities, agent_path)
    pygame.display.update()

# Main game loop (optional for interaction after training)
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.update()

pygame.quit()
