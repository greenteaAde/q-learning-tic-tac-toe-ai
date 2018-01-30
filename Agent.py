import random
import shutil
import itertools
import numpy as np
from Functions import is_finished, encode, decode


class AIagent_RL:
    def __init__(self, restore=False):
        # self.value key : '012301230', value : 1
        self.value = dict()
        if not restore:
            self.init_value()
        else:
            self.restore()

    def init_value(self):

        state_list = itertools.product([0, 1, 2], repeat=9)

        for state in state_list:
            state = list(state)
            done, winner = is_finished(state)
            encoded = encode(state)
            if not done:
                self.value[encoded] = 0
            elif winner == 1:
                self.value[encoded] = 1
            elif winner == 2:
                self.value[encoded] = -1
            else:
                self.value[encoded] = 0

    def policy(self, state, turn, available, epsilon=0.08):
        maxvalue = -99999
        minvalue = 99999
        action_list = []

        if np.random.rand(1) < epsilon:
            action_list = available
        else:
            if turn == 1:
                for i in available:
                    state[i] = turn
                    state = encode(state)
                    if self.value[state] > maxvalue:
                        action_list = []
                        maxvalue = self.value[state]
                        action_list.append(i)
                    elif self.value[state] == maxvalue:
                        action_list.append(i)
                    state = decode(state)
                    state[i] = 0
            else:
                for i in available:
                    state[i] = turn
                    state = encode(state)
                    if self.value[state] < minvalue:
                        action_list = []
                        minvalue = self.value[state]
                        action_list.append(i)
                    elif self.value[state] == minvalue:
                        action_list.append(i)
                    state = decode(state)
                    state[i] = 0

        return random.choice(action_list)

    def save(self):
        shutil.copy2("./data/save.dat", "./data/save_backup.dat")
        with open("./data/save.dat", 'w') as f:
            for key, value in self.value.items():
                f.write(key + ' ' + str(value) + '\n')
        print("saved!")

    def restore(self):
        with open("./data/save_bst.dat", 'r') as f:
            for line in f:
                tmp = line.split()
                key = tmp[0]
                value = float(tmp[1])
                self.value[key] = value
        print("restored!")


class AIagent_Base:
    def policy(self, state, turn, available, epsilon=0):
        action_list = []

        for i in available:
            state[i] = turn
            done, winner = is_finished(state)
            state[i] = 0
            if done:
                action_list.append(i)
        if len(action_list) == 0:
            action_list = available

        return random.choice(action_list)


class Human_agent:
    def policy(self, state, turn, available, epsilon=0):
        while True:
            ret = int(input("input [0 1 2 / 3 4 5 / 6 7 8] : "))
            if ret in available:
                break
        return ret
