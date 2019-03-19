import numpy as np
import copy
import os
import pickle
from time import sleep
import network
import state
import game


def load_game(gameid):
    filename = 'replays/game_{0:d}'.format(gameid)
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)


def q(state, action, w):
    newstate = copy.deepcopy(state)
    newstate.step(action)
    return v(newstate, action, w)


def get_feat_nn(state, action):
    my_map = state.return_maps()
    summap0 = my_map[0][1] 
    summap0 = summap0.flatten()
    summap1 = my_map[1][1].flatten()
    n_feat = 2 * len(summap0) + 9
    x = np.zeros(n_feat)
    x[0] = np.min([np.sum(my_map[0][0].flatten()), 10]) / 10
    x[1] = np.min([np.sum(my_map[1][0].flatten()), 10]) / 10
    x[2] = np.sum(my_map[0][1].flatten()) / 10
    x[3] = np.sum(my_map[1][1].flatten()) / 10
    x[4] = np.sum(my_map[0][3].flatten()) / 10 
    x[5] = np.sum(my_map[1][3].flatten()) / 10
    distancesum = 0
    otherhome = state.players[1][0][0].position
    for i, unit in enumerate(state.players[0][1]):
        d = (np.abs(unit.position[0] - otherhome[0]) + np.abs(unit.position[1] - otherhome[1]))
        if d == 0:
            distancesum = 1e3
        else:
            distancesum += unit.size / d
    x[6] = distancesum / 10
    distancesum = 0
    otherhome = state.players[0][0][0].position
    for i, unit in enumerate(state.players[1][1]):
        d = (np.abs(unit.position[0] - otherhome[0]) + np.abs(unit.position[1] - otherhome[1]))
        if d == 0:
            distancesum = 1e3
        else:
            distancesum += unit.size / d
    x[7] = distancesum / 10
    x[8] = action[0]
    x[9:9 + len(summap0)] = summap0 / 5
    x[9 + len(summap1):] = summap1 / 5
    return x


def v(state, action, net):

    x = get_feat_nn(state, action)

    value = net.feedforward(x[:, None])[0]
    if state.done:
        if state.victory == 0.0:
            return 0.5, x
        else:
            return (state.victory + 1) * 0.5, x
    return value, x


def view_replay(gameid, net):
    replay, victory = load_game(gameid)
    n_turns = len(replay)
    print(n_turns)
    print(replay[0])
    print(len(replay))
    p = 0
    for state, action in replay:
        state.visualize()
        print(action)
        x = get_feat_nn(state, action)
        
        print(net.feedforward(x[:, None])[0])
        print(state.done)
        p += 1
        print(p)
        sleep(1.5)
    print(victory)


def train_nn(net):
    for i in range(2):
        print(i)
        eps = 0.7
        lens = np.zeros(100)
        accuracy = []
        victories = []
        x = []
        y = []
        for gameid in range(100):
            my_game = game.Game(n_pix, q)
            my_game.aivsai(net, net, eps=eps)
            # replay, victory, _ = load_game(gameid)
            replay = my_game.replay
            victory = my_game.state.victory
            # print(my_game.state.victory)
            # print(my_game.turn)
            # print(len(replay))
            n_replay = len(replay)
            lens[gameid] = n_replay
            victory = (victory + 1) * 0.5
            victories.append(victory)
            v_pred = np.zeros(len(replay))
            # x_arr = 
            #history = np.min([20, len(replay)])
            # discount = 1.1
            for i, stac in enumerate(replay):
                state, action = stac
                x1 = get_feat_nn(state, action)
                # y.append(np.array([victory]))
                x.append(x1[:, None])
                if state.done:
                    v_pred[i] = (state.victory + 1) * 0.5
                else:
                    v_pred[i] = net.feedforward(x1[:, None])[0]
                

                # w += alpha * discount * (victory_local - val) * x
                # discount = discount * 1.1
            # print(gameid)
            # print(w)
            v_out = TD_lambda(v_pred)
            for i in range(n_replay):
                y.append(np.array([v_out[i]]))
                accuracy.append(np.sqrt((v_out[i] - v_pred[i]) ** 2))
        print('Average error', np.array(accuracy).mean())
        print('Average result', np.array(victories).mean())
        print('Median game length: ', np.median(lens))

        training_data = list(zip(x, y))
        # net = network.Network([len(x1), 50, 20, 1])
        net.SGD(training_data, 10, 20, 0.1)
    filename = 'nn_2_0'
    with open(filename + '.pkl', 'wb') as f:
            pickle.dump(net, f, pickle.HIGHEST_PROTOCOL)
    return net


def TD_lambda(v_pred, gamma=1, lamb=0.9):
    n_pred = len(v_pred)
    v_out = np.zeros_like(v_pred)
    for i in range(n_pred - 1):
        for j in range(n_pred - i - 1):
            v_out[i] += lamb ** (j-1) * gamma ** j * v_pred[i+j]
        v_out[i] *= (1 - lamb)
        v_out[i] += lamb ** (n_pred - i) * v_pred[-1]
    v_out[-1] = v_pred[-1]
    # print(v_out[-1])
    return v_out


def load_nn():
    filename = 'nn_2_0'
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)

n_pix = 4
n_feat = 2 * n_pix ** 2 + 9
# my_state = State(n_pix)

# my_map = my_state.return_maps()
# w0 = [0.6, 2.0, 0.1, 0.1, 0.1, 0.5]
# net = load_nn()
net = network.Network([n_feat, 128, 128, 16, 1])
train_nn(net)
# # w0 = [0.6, 2.0, 0.1, 0.1, 1.5 , 2.5]
for i in range(1):
    view_replay(1, net)