import numpy as np
import copy
import os
import pickle
from time import sleep
import network
import state
import game
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam


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
    # print(x.shape)
    value = net.predict(x[None, :])
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
        
        print(net.predict(x[None, :]))
        print(state.done)
        p += 1
        print(p)
        sleep(1.5)
    print(victory)


def train_nn(net, elo=0.0):
    for i in range(1):
        print(i)
        eps = 0.3
        n_games = 3000
        lens = np.zeros(n_games)
        accuracy = []
        victories = []
        x = []
        y = []
        for gameid in range(n_games):
            my_game = game.Game(n_pix, q, randomize=True, rand_frac=0.9)
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
                x.append(x1[:])
                if state.done:
                    v_pred[i] = (state.victory + 1) * 0.5
                else:
                    v_pred[i] = net.predict(x1[None, :])
                

                # w += alpha * discount * (victory_local - val) * x
                # discount = discount * 1.1
            # print(gameid)
            # print(w)
            v_out = TD_lambda(v_pred)
            for i in range(n_replay):
                # y.append(np.array([v_out[i]]))
                y.append(v_out[i])
                accuracy.append(np.sqrt((v_out[i] - v_pred[i]) ** 2))
        
        y = np.array(y)
        x = np.array(x)
        # training_data = list(zip(x, y))
        # net = network.Network([len(x1), 50, 20, 1])
        # net.SGD(training_data, 10, 20, 0.1)
        net.fit(x, y, epochs=10)
        print('Average error', np.array(accuracy).mean())
        print('Average result', np.array(victories).mean())
        print('Median game length: ', np.median(lens))
    filename = 'nn_3_0'
    with open(filename + '.pkl', 'wb') as f:
                pickle.dump(net, f, pickle.HIGHEST_PROTOCOL)
    
    mod = [elo, net]
    mod2 = load_ref('ref_best')
    elo_new, _, mean_score = compare_models(mod, mod2, n_exp=500, ref=True)
    mod = [elo_new, net]

    print("New elo: ", elo_new)
    if (elo_new > elo + 15) and (mean_score > 0.53):
        save_ref(mod, 'ref_' + str(int(elo)))
        save_ref(mod, 'ref_best')
        return [elo_new, net]
    else:
        return load_ref('ref_best')


def compare_models(mod1, mod2, n_exp=100, ref=False):
    elo1, model1 = mod1
    elo2, model2 = mod2
    vic = np.zeros(n_exp)
    K = 10
    eps = 0.1
    for i in range(n_exp):
        if i < n_exp // 2:
            gm = game.Game(n_pix, q, randomize=True, rand_frac=0.2)
            gm.aivsai(model1, model2, eps=eps)
            vic[i] = (gm.state.victory + 1) * 0.5
        else:
            gm = game.Game(n_pix, q, randomize=True, rand_frac=0.2)
            gm.aivsai(model2, model1, eps=eps)
            vic[i] = 1 - (gm.state.victory + 1) * 0.5

        exp1, exp2 = get_exp(elo1, elo2)
        
        elo1 = elo1 + K * (vic[i] - exp1)
        if not ref:
            elo2 = elo2 + K * ((1 - vic[i]) - exp2)
    print('Mean score: ', np.mean(vic))
    return elo1, elo2, np.mean(vic)


def get_exp(e1, e2):
    q1 = 10 ** (e1 / 400)
    q2 = 10 ** (e2 / 400)
    qsum = q1 + q2
    e1 = q1 / qsum
    e2 = q2 / qsum
    return e1, e2 


def TD_lambda(v_pred, gamma=0.99, lamb=0.9):
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


# def build_model(input_size, output_size):
#     model = Sequential()
#     model.add(Dense(128, input_dim=input_size, activation='relu'))
#     model.add(Dense(52, activation='relu'))
#     model.add(Dense(output_size, activation='linear'))
#     model.compile(loss='mse', optimizer=Adam())
#     return model

# def train_model(training_data, model):
#     print(training_data[0])
#     X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
#     y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
#     # model = build_model(input_size=len(X[0]), output_size=len(y[0]))
#     model.fit(X, y, epochs=10)
#     return model


def load_nn():
    filename = 'nn_3_0'
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_ref(mod, name='ref'):
    with open(name + '.pkl', 'wb') as f:
            pickle.dump(mod, f, pickle.HIGHEST_PROTOCOL)


def load_ref(name='ref'):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


n_pix = 4
n_feat = 2 * n_pix ** 2 + 9
learning_rate = 0.001

# model = Sequential()
# model.add(Dense(128, input_dim=n_feat, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(52, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

# model = load_nn()
elo, model = load_ref(name='ref')
# mod = train_nn(model, elo=elo)
# save_ref(mod, name='ref')
# print("Elo: ", mod[0])

mod = [1000, model]

mod2 = load_ref('ref_1000')

print(compare_models(mod, mod2, n_exp=300, ref=True))
# save_ref(mod, name='ref_1000')

for i in range(1):
    view_replay(1, model)

# my_game = game.Game(n_pix, q)
# my_game.vs_ai(model)

