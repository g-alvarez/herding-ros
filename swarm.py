#!/usr/bin/env python3
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

# The generalized function parameters
PHI_BETA = 5.3
PHI_C = 0.4

# The gamma function parameters
GAMMA_ALPHA = 2.3

# The attraction function parameters
ATTRACTION_A = 2.5
# The repulsion function parameters
REPULSION_B = 10

def plot_positions(positions):
  """Plot the positions of each robot in a 200x200 figure.
  
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

  If the value of position j is equal to 1 then it is a neighbor.
  """
  j = 0
  for elem in adj_matrix[i]:
    if elem == 1:
      yield j
    j += 1

def convert_range(x):
  """Convert x value from range [0, 1] to range [-100, 100].
  """
  old_range = OLD_MAX - OLD_MIN  
  new_range = MAX - MIN  
  return (((x - OLD_MIN) * new_range) / old_range) + MIN

def phi(value, beta=PHI_BETA, c=PHI_C):
  """The generalized function.

  The formula:
    phi(||y||) = exp(-||y|| ^ beta / c)

  Where:
    - beta >= 1
    - c > 0
  """
  return math.exp(-1 * (value ** beta) / c)

def gamma(pos_i, pos_j, alpha=GAMMA_ALPHA):
  """The gamma function.

  The formula:
    gamma(i, j) = 1 / ||Xi - Xj|| ^ alpha

  Where:
    - alpha > 0
    - Xi, Xj are 2D vectors
  """
  return 1 / (np.linalg.norm(pos_i - pos_j) ** alpha)

def attraction_function(vector, a=ATTRACTION_A):
  """The attraction function.

  The formula:
    Ga(||y||) = a * (1 - phi(||y||)) 

  Where:
    - a > 0
    - y is a 2D vector
    - phi is the generalized function
  """
  return a * (1 - phi(np.linalg.norm(vector)))

def repulsion_function(vector, b=REPULSION_B):
  """The repulsion function.

  The formula:
    Gr(||y||) = b * phi(||y||) 

  Where:
    - b > 0
    - y is a 2D vector
    - phi is the generalized function
  """
  return b * phi(np.linalg.norm(vector))

def interaction_function(vector):
  """The interaction function.

  The formula:
    g(y) = y / ||y|| * [( Gr(||y||) - Ga(||y||) )]
    
  Where:
    - y is a 2D vector
    - Gr is the repulsion function
    - Ga is the attraction function
  """
  return vector / np.linalg.norm(vector) * (repulsion_function(vector) - attraction_function(vector))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Swarm algorithm test.')
  parser.add_argument('-n', '--robots', type=int, default=2, choices=range(2, 11), help='The number of robots')
  parser.add_argument('-T', '--rate', type=int, default=5, choices=range(1, 11), help='The rate of positions plots')
  parser.add_argument('-v', '--verbose', action="store_true", help='Verbose mode shows program variables state')

  args = parser.parse_args()
  n = args.robots
  
  # The adjancency matrix, all robots are connected
  adj_matrix = np.ones((n, n), dtype=int)
  np.fill_diagonal(adj_matrix, 0)

  # The position (x, y) coordinates of robot i for i in 0..n-1
  positions = convert_range(np.random.rand(n, 2))

  if args.verbose:
    print("Adjacency matrix:")
    print(adj_matrix)
    print()

  while True:
    # Plot the positions of all robots
    plot_positions(positions)

    # Calculate the new positions
    for i in range(n):
      if args.verbose:
        print("Robot [{}] - {}".format(i, positions[i]))

      sum_gamma_interaction_function = np.zeros(2)
      sum_gamma = np.zeros(2)
      # Get the neighbors of robot i
      for j in get_neighbors(i, adj_matrix):
        sum_gamma_interaction_function += gamma(positions[i], positions[j]) * interaction_function(positions[i] - positions[j])
        sum_gamma += gamma(positions[i], positions[j])

      # Update the position of robot i with the accumulated values
      positions[i] += sum_gamma_interaction_function / sum_gamma
      # Make sure coordinates are inside the map
      positions[i][0] = positions[i][0] if positions[i][0] >= MIN else MIN - positions[i][0] % MIN
      positions[i][0] = positions[i][0] if positions[i][0] <= MAX else MAX - positions[i][0] % MAX
      positions[i][1] = positions[i][1] if positions[i][1] <= MAX else MAX - positions[i][1] % MAX
      positions[i][1] = positions[i][1] if positions[i][1] >= MIN else MIN - positions[i][1] % MIN

    if args.verbose:
      print("Sleeping for {} seconds...\n".format(1 / args.rate))
    time.sleep(1 / args.rate)
