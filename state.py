import numpy as np
import copy
import os
import pickle
import units as units
from time import sleep


class State():
    def __init__(self, n_pix, players=None):
        self.n_pix = n_pix
        self.done = False
        if players is None:
            self.players = [
                [[units.Center((0, 0))], [units.Unit((0, 1))]],
                [[units.Center((self.n_pix - 1, self.n_pix - 1))],
                 [units.Unit((self.n_pix - 1, self.n_pix - 2))]]
            ]
        else:
            self.players = players

    def return_maps(self):
        my_map = np.zeros((len(self.players), len(self.players[0]) + 2, self.n_pix, self.n_pix))
        my_map[:, len(self.players[0])] = -1
        for i, player in enumerate(self.players):
            for j, unit_class in enumerate(player):
                for k, unit in enumerate(unit_class):
                    my_map[i, j, unit.position[0], unit.position[1]] = unit.size
                    my_map[i, len(self.players[0]), unit.position[0], unit.position[1]] = k  # indices
                    if j == 0:
                        my_map[i, len(self.players[0]) + 1, unit.position[0], unit.position[1]] = unit.res
        return my_map

    def visualize(self):
        my_map = self.return_maps()
        print('Money %i and %i' % (np.sum(my_map[0][3].flatten()), np.sum(my_map[1][3].flatten())))
        summap = my_map[0][1] - my_map[1][1] + my_map[0][0] - my_map[1][0]
        print(summap)
        # print(my_map[0][0] - my_map[1][0])
        # print(my_map[0][2])
        # print(my_map[1][2])

    def step(self, action):
        self.players[action[0]][action[1]][action[2]].act(self, action)
        for player in self.players:
            for centre in player[0]:
                centre.work()