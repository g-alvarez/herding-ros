#!/usr/bin/env python3
# coding=utf-8

import numpy as np
import math
import argparse
import time
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from beartype import beartype
from typing import Iterator

matplotlib.use("TkAgg")

# np.random.rand range
OLD_MIN = 0
OLD_MAX = 1

# Plot figure (x,y) axis range
MIN = -100
MAX = 100

# The generalized function parameters
PHI_BETA = 5.3
PHI_C = 0.4

# The gamma function parameter
GAMMA_ALPHA = 2.3

# The attraction function parameter
ATTRACTION_A = 10.0

# The repulsion function parameter
REPULSION_B = 3.0

# The range of visibility
R = 1000

@beartype
def plot_positions(positions: np.ndarray) -> None:
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

@beartype
def get_neighbors(i: int, adj_matrix: np.ndarray) -> Iterator[int]:
  """Get the neighbors of robot i, using the adjancency matrix.

  If the value of position j is equal to 1 then it is a neighbor.
  """
  j = 0
  for elem in adj_matrix[i]:
    if elem == 1:
      yield j
    j += 1

@beartype
def update_adj_matrix(positions: np.ndarray, adj_matrix: np.ndarray) -> None:
  """Update the adjacency matrix with the current position of all robots.

  The formula:
    eij(t) = 1 <=> ||Xi(t) - Xj(t)|| <= r, otherwise 0
    aij(t) = 1 <=> eij(t) = 1, otherwise 0

  Where:
    eij is an edge between robot i and robot j
    aij is an entry in the adjacency matrix
  """
  n = len(adj_matrix)
  for i in range(n):
    for j in range(i+1, n):
      distance = np.linalg.norm(positions[i] - positions[j])
      adj_matrix[i][j] = 1 if distance <= R else 0
      adj_matrix[j][i] = 1 if distance <= R else 0

@beartype
def convert_range(vector: np.ndarray) -> np.ndarray:
  """Convert vector values from range [0, 1] to range [-100, 100].
  """
  old_range = OLD_MAX - OLD_MIN  
  new_range = MAX - MIN  
  return (((vector - OLD_MIN) * new_range) / old_range) + MIN

@beartype
def phi(value: float, beta: float = PHI_BETA, c: float = PHI_C) -> float:
  """The generalized function.

  The formula:
    phi(||y||) = exp(-||y|| ^ beta / c)

  Where:
    - beta >= 1
    - c > 0
  """
  return math.exp(-1 * (value ** beta) / c)

@beartype
def gamma(pos_i: np.ndarray, pos_j: np.ndarray, alpha: float = GAMMA_ALPHA) -> float:
  """The gamma function.

  The formula:
    gamma(i, j) = 1 / ||Xi - Xj|| ^ alpha

  Where:
    - alpha > 0
    - Xi, Xj are 2D vectors
  """
  return 1 / (np.linalg.norm(pos_i - pos_j) ** alpha)

@beartype
def attraction_function(vector: np.ndarray, a: float = ATTRACTION_A) -> float:
  """The attraction function.

  The formula:
    Ga(||y||) = a * (1 - phi(||y||)) 

  Where:
    - a > 0
    - y is a 2D vector
    - phi is the generalized function
  """
  return a * (1 - phi(np.linalg.norm(vector)))

@beartype
def repulsion_function(vector: np.ndarray, b: float = REPULSION_B) -> float:
  """The repulsion function.

  The formula:
    Gr(||y||) = b * phi(||y||) 

  Where:
    - b > 0
    - y is a 2D vector
    - phi is the generalized function
  """
  return b * phi(np.linalg.norm(vector))

@beartype
def interaction_function(vector: np.ndarray) -> np.ndarray:
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
  parser.add_argument('-n', '--robots', type=int, default=2, choices=range(2, 101), help='the number of robots')
  parser.add_argument('-r', '--rate', type=int, default=5, choices=range(1, 11), help='the rate of plots')
  parser.add_argument('-v', '--verbose', action="store_true", help='shows program state step by step')

  args = parser.parse_args()
  n = args.robots
  
  # The adjancency matrix, all robots are connected
  adj_matrix = np.ones((n, n), dtype=int)
  np.fill_diagonal(adj_matrix, 0)

  # The position (x, y) coordinates of robot i for i in 0..n-1
  positions = convert_range(np.random.rand(n, 2))

  while True:
    # Plot the positions of all robots
    plot_positions(positions)

    # Update the adjacency matrix with current positions
    update_adj_matrix(positions, adj_matrix)

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
      if np.linalg.norm(sum_gamma) != 0:
        positions[i] += sum_gamma_interaction_function / sum_gamma
      # Make sure coordinates are inside the map
      positions[i][0] = positions[i][0] if positions[i][0] >= MIN else MIN - positions[i][0] % MIN
      positions[i][0] = positions[i][0] if positions[i][0] <= MAX else MAX - positions[i][0] % MAX
      positions[i][1] = positions[i][1] if positions[i][1] <= MAX else MAX - positions[i][1] % MAX
      positions[i][1] = positions[i][1] if positions[i][1] >= MIN else MIN - positions[i][1] % MIN

    if args.verbose:
      print("Sleeping for {} seconds...\n".format(1 / args.rate))
    time.sleep(1 / args.rate)
