#!/usr/bin/env python3
# coding=utf-8

# author: Gabriel Álvarez
# email: 781429@unizar.es

import matplotlib.pyplot as plt
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Plot the minimun distances between robots.')
parser.add_argument('-n', '--robots', type=int, required=True, choices=range(2, 26),
                    help='the number of robots')
parser.add_argument('-v', '--verbose', action="store_true",
                    help='shows program state step by step')
parser.add_argument('-o', '--obstacles', type=int, required=True, choices=range(0, 11),
                    help='the number of obstacles')

args = parser.parse_args()
num_robots = args.robots
num_obstacles = args.obstacles

alpha_1_df = pd.read_csv('./csv/good/distances.csv', sep=",", encoding="utf-8")
alpha_2_df = pd.read_csv('./csv/alpha=2.0/distances.csv', sep=",", encoding="utf-8")
alpha_3_df = pd.read_csv('./csv/alpha=3.0/distances.csv', sep=",", encoding="utf-8")

# Calculate min distance in batches of number of robots
min_dists_robots_alpha_1 = []
max_dists_robots_alpha_1 = []
min_dists_robots_alpha_2 = []
max_dists_robots_alpha_2 = []
min_dists_robots_alpha_3 = []
max_dists_robots_alpha_3 = []
for i in range(0, len(alpha_1_df), num_robots):
  min_dist_robots_alpha_1 = alpha_1_df[i:i+num_robots][['neighbor_' +
                                                          str(i) for i in range(num_robots)]].min().min()
  max_dist_robots_alpha_1 = alpha_1_df[i:i+num_robots][['neighbor_' +
                                                          str(i) for i in range(num_robots)]].max().max()
  min_dist_robots_alpha_2 = alpha_2_df[i:i+num_robots][['neighbor_' +
                                                          str(i) for i in range(num_robots)]].min().min()
  max_dist_robots_alpha_2 = alpha_2_df[i:i+num_robots][['neighbor_' +
                                                          str(i) for i in range(num_robots)]].max().max()
  min_dist_robots_alpha_3 = alpha_3_df[i:i+num_robots][['neighbor_' +
                                                          str(i) for i in range(num_robots)]].min().min()
  max_dist_robots_alpha_3 = alpha_3_df[i:i+num_robots][['neighbor_' +
                                                          str(i) for i in range(num_robots)]].max().max()
  if args.verbose:
    print("Current batch:", i, i+num_robots)
    print("- Alpha = 1.0 Minimun distance robots:", min_dist_robots_alpha_1)
    print("- Alpha = 1.0 Maximun distance leader:", max_dist_robots_alpha_1)
    print("- Alpha = 2.0 Minimun distance robots:", min_dist_robots_alpha_2)
    print("- Alpha = 2.0 Maximun distance leader:", max_dist_robots_alpha_2)
    print("- Alpha = 3.0 Minimun distance robots:", min_dist_robots_alpha_3)
    print("- Alpha = 3.0 Maximun distance leader:", max_dist_robots_alpha_3)
  min_dists_robots_alpha_1.append(min_dist_robots_alpha_1)
  max_dists_robots_alpha_1.append(max_dist_robots_alpha_1)
  min_dists_robots_alpha_2.append(min_dist_robots_alpha_2)
  max_dists_robots_alpha_2.append(max_dist_robots_alpha_2)
  min_dists_robots_alpha_3.append(min_dist_robots_alpha_3)
  max_dists_robots_alpha_3.append(max_dist_robots_alpha_3)

x = list(range(int(len(alpha_1_df) / num_robots)))

plt.plot(x, min_dists_robots_alpha_1,
          label="Alpha = 1.0 Minimun distance robots")
plt.plot(x, max_dists_robots_alpha_1,
          label="Alpha = 1.0 Maximun distance robots")
plt.plot(x, min_dists_robots_alpha_2,
          label="Alpha = 2.0 Minimun distance robots")
plt.plot(x, max_dists_robots_alpha_2,
          label="Alpha = 2.0 Maximun distance robots")
plt.plot(x, min_dists_robots_alpha_3,
          label="Alpha = 3.0 Minimun distance robots")
plt.plot(x, max_dists_robots_alpha_3,
          label="Alpha = 3.0 Maximun distance robots")
plt.legend()
plt.savefig('./img/alpha_distances.png')
