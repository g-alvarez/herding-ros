#!/usr/bin/env python3
# coding=utf-8

from matplotlib import animation
from beartype import beartype
from typing import Iterator
import math
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

DELTA_T = 0.03

# np.random.rand range
OLD_MIN = 0.0
OLD_MAX = 1.0

# Plot figure (x,y) axis range
MIN = -2.0
MAX = 2.0

# The generalized function parameters
PHI_BETA = 4.0
PHI_C = 0.02

# The gamma function parameter
GAMMA_ALPHA = 1.0

# The attraction function parameter
ATTRACTION_A = 4.0

# The repulsion function parameter
REPULSION_B = 0.4

# The leader speed parameter
LEADER_SPEED = 0.005

# The leader node attraction function parameter
LEADER_ATTRACTION_A = 20.0

# The obstacle node repulsion function parameter
OBSTACLE_REPULSION_B = 20.0

# The range of visibility
R = 100.0

# The angular speed
DELTA_ANG = 0.5

# The radius of the circular trajectory
RADIUS = 1.0

sin = lambda degs: math.sin(math.radians(degs))
cos = lambda degs: math.cos(math.radians(degs))

@beartype
def circular_path(x: float, y: float, delta_angle: float = DELTA_ANG, 
                  radius: float = RADIUS) -> Iterator[tuple]:
  """Generate coords of a circular path every delta angle degrees.
  """
  angle = 0
  while True:
    yield (x + radius * cos(angle), y + radius * sin(angle))
    angle = (angle + delta_angle) % 360

@beartype
def plot_positions(n: int, positions: np.ndarray) -> None:
  """Plot the positions of each robot in a 200x200 figure.
  
  The x and y axis limits:
    - x [-100, 100]
    - y [-100, 100]
  """
  plt.clf()
  plt.xlim(MIN, MAX)
  plt.ylim(MIN, MAX)
  i = 0
  for pos in positions:
    # Robots
    if i < n:
      plt.plot(pos[0], pos[1], "bo")
    # Leader
    if i == n:
      plt.plot(pos[0], pos[1], "ko")
    # Obstacles
    elif i > n:
      plt.plot(pos[0], pos[1], "go")
    i += 1
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
def update_adj_matrix(n: int, positions: np.ndarray, adj_matrix: np.ndarray) -> None:
  """Update the adjacency matrix with the current position of all robots.

  The formula:
    eij(t) = 1 <=> ||Xi(t) - Xj(t)|| <= r, otherwise 0
    aij(t) = 1 <=> eij(t) = 1, otherwise 0

  Where:
    eij is an edge between robot i and robot j
    aij is an entry in the adjacency matrix
  """
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
def interaction_function(vector: np.ndarray, neighbor: int, n: int) -> np.ndarray:
  """The interaction function.

  The formula:
    g(y) = y / ||y|| * [( Gr(||y||) - Ga(||y||) )]
    
  Where:
    - y is a 2D vector
    - Gr is the repulsion function
    - Ga is the attraction function
  """
  # Neighbor is leader node
  if neighbor == n:
    return vector / np.linalg.norm(vector) * (repulsion_function(vector) - attraction_function(vector, a=LEADER_ATTRACTION_A))
  elif neighbor > n:
    return vector / np.linalg.norm(vector) * (repulsion_function(vector, b=OBSTACLE_REPULSION_B) - attraction_function(vector))
  return vector / np.linalg.norm(vector) * (repulsion_function(vector) - attraction_function(vector))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Swarm algorithm test.')
  parser.add_argument('-n', '--robots', type=int, default=2, choices=range(2, 25), 
                      help='the number of robots')
  parser.add_argument('-r', '--rate', type=int, default=5, choices=range(1, 11), 
                      help='the rate of plots')
  parser.add_argument('-v', '--verbose', action="store_true",
                      help='shows program state step by step')
  parser.add_argument('-m', '--motion', choices=["circ", "rect"], default="rect",
                      help='leader node motion type')

  args = parser.parse_args()
  n = args.robots
  motion = args.motion

  # The position (x, y) coordinates of robot i for i in 0..n-1
  positions = convert_range(np.random.rand(n+1, 2))
  positions_next = np.zeros((n, 2))
  positions[n] = np.array([0.0, 0.0])

  if motion == "circ":
    c = circular_path(positions[n][0], positions[n][1])
    positions[n] = next(c)

  # The obstacle coordinates, a horizontal line y = 10.0
  xaxis = np.linspace(-1.0, 1.0, 4)
  yaxis = np.full(len(xaxis), 1.0, dtype=float)
  obstacles = np.array(list(zip(xaxis,yaxis)))

  positions = np.concatenate((positions, obstacles), axis=0)

  # The adjancency matrix, all robots are connected
  adj_matrix = np.ones((len(positions), len(positions)), dtype=int)
  np.fill_diagonal(adj_matrix, 0)

  k = 0
  while True:
    # Plot the positions of all robots
    plot_positions(n, positions)

    # Update the adjacency matrix with current positions
    update_adj_matrix(n, positions, adj_matrix)

    # Calculate the new positions
    for i in range(n):
      positions_next[i] = positions[i]
      if args.verbose:
        print("Robot [{}] - {}".format(i, positions[i]))

      sum_gamma_interaction_function = np.zeros(2)
      sum_gamma = 0.0
      # Get the neighbors of robot i
      for j in get_neighbors(i, adj_matrix):
        gamma_res = gamma(positions[i], positions[j])
        sum_gamma_interaction_function += gamma_res * \
            interaction_function(positions[i] - positions[j], j, n)
        sum_gamma += gamma_res

      # Update the position of robot i with the accumulated values
      if np.linalg.norm(sum_gamma) != 0:
        positions_next[i] += sum_gamma_interaction_function / sum_gamma * DELTA_T

    # Update the robot position
    for i in range(n):
      positions[i] = positions_next[i]
      
    # Start moving the leader after 10 iterations
    if k > 10:
      if motion == "circ":
        positions[n] = next(c)
      else:
        positions[n] += LEADER_SPEED

    k += 1
    if args.verbose:
      print("Sleeping for {} seconds...\n".format(1 / args.rate))
    time.sleep(1 / args.rate)
