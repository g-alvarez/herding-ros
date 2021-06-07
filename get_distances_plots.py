#!/usr/bin/env python3
# coding=utf-8

import matplotlib.pyplot as plt
import pandas as pd
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Plot the minimun distances between robots.')
  parser.add_argument('-n', '--robots', type=int, required=True, choices=range(2, 26),
                      help='the number of robots')
  parser.add_argument('-v', '--verbose', action="store_true",
                      help='shows program state step by step')
  parser.add_argument('-o', '--obstacles', type=int, required=True, choices=range(0, 11),
                      help='the number of obstacles')
  parser.add_argument('-t', '--type', type=str, choices=["good", "regular", "bad"],
                      help='the type of run')

  args = parser.parse_args()
  num_robots = args.robots
  num_obstacles = args.obstacles

  df = pd.read_csv('./csv/' + (args.type + '/' if args.type else '') + 'distances.csv', sep=",", encoding="utf-8")

  # Calculate min distance in batches of number of robots
  min_dists_robots = []
  min_dists_leader = []
  min_dists_obstacles = []
  for i in range(0, len(df), num_robots):
    min_dist_robots = df[i:i+num_robots][['neighbor_' +
                                    str(i) for i in range(num_robots)]].min().min()
    min_dist_leader = df[i:i+num_robots][['leader']].min().min()
    min_dist_obstacles = df[i:i+num_robots][['obstacle_' +
                                        str(i) for i in range(num_obstacles)]].min().min()
    if args.verbose:
      print("Current batch:", i, i+num_robots)
      print("- Minimun distance robots:", min_dist_robots)
      print("- Minimun distance leader:", min_dist_leader)
      print("- Minimun distance obstacles:", min_dist_obstacles)
    min_dists_robots.append(min_dist_robots)
    min_dists_leader.append(min_dist_leader)
    min_dists_obstacles.append(min_dist_obstacles)

  x = list(range(int(len(df) / num_robots)))

  plt.plot(x, min_dists_robots, label="Robots")
  plt.plot(x, min_dists_leader, label="Leader")
  plt.plot(x, min_dists_obstacles, label="Obstacles")
  plt.legend()
  plt.savefig('./img/' + (args.type + '/' if args.type else '') + 'distances.png')
