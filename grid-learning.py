#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

'''
Direction constants used to move through grid world.
'''
UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3

'''
Grid world of given square size w/ given probability of successfully moving as 
intended through it. (1 - success_probability) chace to move randomly instead.

Inital state : 0, 0 (bottom left)
Goal state   : n-1, n-1 (top right)
'''
class GridWorld:

    def __init__(self, grid_size, success_probability):

        # min/max grid coordinate values
        self.grid_min = 0
        self.grid_max = grid_size - 1

        # probability of taking the action intended when moving
        self.success_probability = success_probability

        # set up initial state
        self.x = 0
        self.y = 0

        # set up goal state
        self.goal_x = self.grid_max
        self.goal_y = self.grid_max

    # attempt to move in a given direction 
    def move(self, direction):

        # move in a random direction with probability (1 - success_probability)
        if np.random.binomial(1, self.success_probability) == 0:
            direction = int(np.random.uniform(0, 4))

        # move up if possible
        if (direction == UP): 
            if (self.y < self.grid_max): self.y += 1

        # move down if possible
        elif (direction == DOWN): 
            if (self.y > self.grid_min): self.y -= 1
        
        # move right if possible
        elif (direction == RIGHT): 
            if (self.x < self.grid_max): self.x += 1
        
        # move left if possible
        elif (direction == LEFT): 
            if (self.x > self.grid_min): self.x -= 1

        # goal state reached?
        if (self.x == self.grid_max) and (self.y == self.grid_max): 

            # return direction taken, reward, and True to indicate goal reached            
            return (direction, 1, True)

        # return direction taken, reward, and False to indicate goal not reached
        return (direction, 0, False)


'''
Navigate given GridWorld using Q-Learning algorithm and plot learning results
'''
def q_learning(grid_world, p, alpha, gamma, episodes):

    # grid size
    n = grid_world.grid_max + 1

    # matrix of dictionaries - Q(s, a) values
    Q_vals = [[{UP : 0, 
                DOWN : 0, 
                RIGHT : 0, 
                LEFT : 0} for i in range(n)] for j in range(n)]

    # learning results to plot
    learning_results = [0]*episodes

    # run episodes
    for i in range(episodes):

        # print every 100 iterations to keep track of progress
        if i % 100 == 0: print("Q-Learning: Episode " + str(i))

        # reset grid world state to initial state
        grid_world.x = 0
        grid_world.y = 0

        # episode variables
        episode_reward = 0
        episode_goal_reached = False
        episode_moves = 0

        # loop until goal reached
        while (not episode_goal_reached):

            # get current grid world state
            x = grid_world.x
            y = grid_world.y

            # determine best direction to take
            direction = max(Q_vals[x][y], key = lambda key: Q_vals[x][y][key])

            # if there was a 4-way tie for best direction, pick a random one
            if (Q_vals[x][y][direction] == 0): 
                direction = int(np.random.uniform(0, 4))

            # move, obtain direction actually moved, reward, and goal boolean
            direction, move_reward, goal_reached = grid_world.move(direction)

            episode_moves += 1
            episode_reward += move_reward
            episode_goal_reached = goal_reached

            # determine next best direction to take
            next_direction = max(Q_vals[grid_world.x][grid_world.y], 
                                 key = lambda key: 
                                        Q_vals[grid_world.x][grid_world.y][key])

            # determine reward for next best direction
            next_reward = Q_vals[grid_world.x][grid_world.y][next_direction]

            # update Q values
            Q_vals[x][y][direction] += alpha*(move_reward 
                                              + gamma*next_reward 
                                              - Q_vals[x][y][direction])

        # add learning result to learning results list
        learning_results[i] = 1 / episode_moves

    # plot learning results for Q-learning
    plt.plot(learning_results)
    plt.title('Q-Learning Results')
    plt.xlabel('Episode')
    plt.ylabel('1 / moves to goal')
    plt.show()


'''
Navigate given GridWorld using SARSA and plot learning results
'''
def sarsa(grid_world, p, alpha, gamma, episodes):

    # grid size
    n = grid_world.grid_max + 1

    # matrix of dictionaries - Q(s, a) values
    Q_vals = [[{UP : 0, 
                DOWN : 0, 
                RIGHT : 0, 
                LEFT : 0} for i in range(n)] for j in range(n)]

    # learning results to plot
    learning_results = [0]*episodes

    # run episodes
    for i in range(episodes):

        # print every 100 iterations to keep track of progress
        if i % 100 == 0: print("SARSA: Episode " + str(i))

        # reset grid world state to initial state
        grid_world.x = 0
        grid_world.y = 0

        # episode variables
        episode_reward = 0
        episode_goal_reached = False
        episode_moves = 0

        # loop until goal reached
        while (not episode_goal_reached):

            # get current grid world state
            x = grid_world.x
            y = grid_world.y

            # determine best direction to take
            direction = max(Q_vals[x][y], key = lambda key: Q_vals[x][y][key])

            # if there was a 4-way tie for best direction, pick a random one
            if (Q_vals[x][y][direction] == 0): 
                direction = int(np.random.uniform(0, 4))

            # move, obtain direction actually moved, reward, and goal boolean
            direction, move_reward, goal_reached = grid_world.move(direction)

            episode_moves += 1
            episode_reward += move_reward
            episode_goal_reached = goal_reached

            # determine next best direction to take
            next_direction = max(Q_vals[grid_world.x][grid_world.y], 
                                 key = lambda key: 
                                        Q_vals[grid_world.x][grid_world.y][key])

            # determine reward for next best direction
            best_reward = Q_vals[grid_world.x][grid_world.y][next_direction]

            # determine reward for random direction
            random_reward = ((Q_vals[grid_world.x][grid_world.y][UP] 
                              + Q_vals[grid_world.x][grid_world.y][DOWN]
                              + Q_vals[grid_world.x][grid_world.y][RIGHT]
                              + Q_vals[grid_world.x][grid_world.y][LEFT]) / 4)

            # expected reward following policy
            next_reward = p*best_reward + (1-p)*random_reward

            # update Q values
            Q_vals[x][y][direction] += alpha*(move_reward 
                                              + gamma*next_reward 
                                              - Q_vals[x][y][direction])

        # add learning result to learning results list
        learning_results[i] = 1 / episode_moves

    # plot learning results for SARSA
    plt.plot(learning_results)
    plt.title('SARSA Results')
    plt.xlabel('Episode')
    plt.ylabel('1 / moves to goal')
    plt.show()


'''
Define and navigate a grid world using reinforcement learning algorithms
'''
def main():

    # grid world & learning parameters
    n = 50
    p = 0.9
    alpha = 0.1
    gamma = 0.95
    episodes = 10000

    # create grid world
    grid_world = GridWorld(n, p)

    # navigate using Q-learning and output results
    q_learning(grid_world, p, alpha, gamma, episodes)

    # navigate using SARSA and output results
    sarsa(grid_world, p, alpha, gamma, episodes)

if __name__ == '__main__':
    main()
