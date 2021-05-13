#!/usr/bin/env python
# coding=utf-8

import numpy as np
import math
import argparse
import time
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation

matplotlib.use("TkAgg")

OLD_MIN = 0
OLD_MAX = 1
MIN = -100
MAX = 100

PHI_BETA = 5.3
PHI_C = 0.4

GAMMA_ALPHA = 2.3

ATTRACTION_A = 2.5
REPULSION_B = 10

def plot_positions(positions):
  """Plot the positions of each robot.
  
  The x and y axis limits:
    - x [-100, 100]
    - y [-100, 100]
  """
  plt.clf()
  plt.xlim(MIN, MAX)
  plt.ylim(MIN, MAX)
  for pos in positions:
    plt.plot(pos[0], pos[1], "bo")
  plt.draw()
  plt.show(block=False)
  plt.pause(0.01)

def get_neighbors(i, adj_matrix):
  """Get the neighbors of robot i, using the adjancency matrix.

  If the position j is equal to 1 and it is not the current robot, 
  then it is a neighbor 
  """
  neighbors = []
  j = 0
  for elem in adj_matrix[i]:
    if elem == 1 and j != i:
      neighbors.append(j)
    j += 1
  return neighbors

def convert_range(x):
  """Convert x value from [0, 1] to [-100, 100]
  """
  old_range = OLD_MAX - OLD_MIN  
  new_range = MAX - MIN  
  return (((x - OLD_MIN) * new_range) / old_range) + MIN

def phi(value, beta=PHI_BETA, c=PHI_C):
  """Solve quadratic equation via the quadratic formula.

  A quadratic equation has the following form:
  ax**2 + bx + c = 0

  beta >= 1, c > 0
  """
  return math.exp(-1 * (value ** beta) / c)

def gamma(pos_i, pos_j, alpha=GAMMA_ALPHA):
  """Solve quadratic equation via the quadratic formula.

  A quadratic equation has the following form:
  ax**2 + bx + c = 0

  alpha > 0
  """
  return 1 / (np.linalg.norm(pos_i - pos_j) ** alpha)

def attraction_function(vector, a=ATTRACTION_A):
  """Solve quadratic equation via the quadratic formula.

  A quadratic equation has the following form:
  ax**2 + bx + c = 0

  a > 0
  """
  return a * (1 - phi(np.linalg.norm(vector)))

def repulsion_function(vector, b=REPULSION_B):
  """Solve quadratic equation via the quadratic formula.

  A quadratic equation has the following form:
  ax**2 + bx + c = 0

  b > 0
  """
  return b * phi(np.linalg.norm(vector))

def interaction_function(vector):
  """Solve quadratic equation via the quadratic formula.

  A quadratic equation has the following form:
  ax**2 + bx + c = 0

  There always two solutions to a quadratic equation: x_1 & x_2.
  """
  return vector / np.linalg.norm(vector) * (repulsion_function(vector) - attraction_function(vector))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Swarm algorithm test.')
  parser.add_argument('-N', '--robots', type=int, default=2, choices=range(2, 11), help='The number of robots')
  parser.add_argument('-T', '--rate', type=int, default=5, choices=range(1, 11), help='The rate of positions plots')

  args = parser.parse_args()
  n = args.robots
  
  # The adjancency matrix, all robots are connected
  adj_matrix = np.ones((n, n), dtype=int)
  np.fill_diagonal(adj_matrix, 0)
  print("Adjacency matrix:")
  print(adj_matrix)
  print()

  # The position (x, y) coordinate of robot i in 0..n-1
  positions = convert_range(np.random.rand(n, 2))
  print("Initial positions:")
  print(positions)
  print()

  while True:
    # Plot the positions of all robots
    plot_positions(positions)

    # Calculate the new positions
    for i in range(n):
      sum_gamma_interaction_function = np.zeros(2)
      sum_gamma = np.zeros(2)
      # Get the neighbors of robot i
      for j in get_neighbors(i, adj_matrix):
        sum_gamma_interaction_function += gamma(positions[i], positions[j]) * interaction_function(positions[i] - positions[j])
        sum_gamma += gamma(positions[i], positions[j])

      # Update the position of robot i with the accumulated values
      positions[i] += sum_gamma_interaction_function / sum_gamma
      print("Robot [{}] - {}".format(i, positions[i]))

    print("Sleeping for {} seconds...\n".format(1 / args.rate))
    time.sleep(1 / args.rate)
