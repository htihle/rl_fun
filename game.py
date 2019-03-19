import numpy as np
import copy
import os
import pickle
import state as st
from time import sleep


class Game():
    def __init__(self, n_pix, q, state=None):
        self.n_pix = n_pix
        self.q = q
        if state is None:
            self.state = st.State(self.n_pix)
        else:
            self.state = state
        self.turn = 1
        
        self.replay = []

    def vs_ai(self, net):
        walking = {
            "s": 0,
            "d": 1,
            "w": 2,
            "a": 3,
        }
        while (not self.state.done):
            self.state.visualize()
            action = [0]
            action.append(int(input('0 for cities, 1 for units: ')))
            if (action[1] == 0):
                print('You have %s resources' % self.state.players[0][0][0].res)
                action.append(0)
                action.append(int(input('0 for growth, 1 for unit: ')))
            elif (action[1] == 1):
                for i, unit in enumerate(self.state.players[action[0]][1]):
                    print('You have unit at (%i, %i) with size %i, this is unit %i in %i' % (
                        unit.position[0], unit.position[1], unit.size, i + 1, len(self.state.players[action[0]][1])))
                    my_input = input('wasd to move or f to skip ')
                    if (my_input == 'f'):
                        pass
                    else:
                        action.append(i)
                        action.append(walking[my_input])
                        break
            self.replay.append([copy.deepcopy(self.state), action])
            self.state.step(action)

            q_min = 1e8
            for i, unittype in enumerate(self.state.players[1]):
                for j, unit in enumerate(unittype):
                    for a in range(unit.n_actions):
                        current_q, _ = self.q(self.state, [1, i, j, a], net)
                        if current_q < q_min:
                            q_min = current_q
                            action = [1, i, j, a]
            print(q_min)
            self.replay.append([copy.deepcopy(self.state), action])
            self.state.step(action)
            self.turn += 1
            if self.turn > 150:
                self.state.done = True
                self.state.victory = 0.0
        if self.state.done:
            self.save_game()

    def aivsai(self, w1, w2, eps=0.2):

        while (not self.state.done):
            if np.random.rand() < eps:
                i = np.random.randint(2)
                if len(self.state.players[0][i]) > 0:
                    j = np.random.randint(len(self.state.players[0][i]))
                else:
                    i = 1 - i
                    j = np.random.randint(len(self.state.players[0][i]))
                a = np.random.randint(self.state.players[0][i][j].n_actions)
                action = [0, i, j, a]
            else:
                q_max = -1e8
                for i, unittype in enumerate(self.state.players[0]):
                    for j, unit in enumerate(unittype):
                        for a in range(unit.n_actions):
                            current_q, _ = self.q(self.state, [0, i, j, a], w1)
                            if current_q > q_max:
                                q_max = current_q
                                action = [0, i, j, a]
                # print(q_max)
            self.state.step(action)
            self.replay.append([copy.deepcopy(self.state), action])
            self.turn += 1
            if self.turn > 300:
                self.state.done = True
                self.state.victory = 0.0

            if np.random.rand() < eps:
                i = np.random.randint(2)
                if len(self.state.players[1][i]) > 0:
                    j = np.random.randint(len(self.state.players[1][i]))
                else:
                    i = 1 - i
                    j = np.random.randint(len(self.state.players[1][i]))
                a = np.random.randint(self.state.players[1][i][j].n_actions)
                action = [1, i, j, a]
            else:
                q_min = 1e8
                for i, unittype in enumerate(self.state.players[1]):
                    for j, unit in enumerate(unittype):
                        for a in range(unit.n_actions):
                            current_q, _ = self.q(self.state, [1, i, j, a], w2)
                            if current_q < q_min:
                                q_min = current_q
                                action = [1, i, j, a]
                # print(q_max)
            self.state.step(action)
            self.replay.append([copy.deepcopy(self.state), action])
            self.turn += 1
            if self.turn > 300:
                self.state.done = True
                self.state.victory = 0.0
        if self.state.done:
            self.replay.append([copy.deepcopy(self.state), action])
            self.save_game()

    def save_game(self):
        all_info = [self.replay, self.state.victory]
        gameid = 0
        while os.path.isfile(os.path.join(
                'replays/game_{0:d}.pkl'.format(gameid))):
            gameid += 1
        # print('Current game_id: %i' % gameid)
        # filename = 'replays/game_{0:d}'.format(gameid)
        filename = 'replays/game_1'
        with open(filename + '.pkl', 'wb') as f:
            pickle.dump(all_info, f, pickle.HIGHEST_PROTOCOL)
