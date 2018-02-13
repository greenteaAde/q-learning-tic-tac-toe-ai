import random
import shutil
import itertools
import numpy as np
from Functions import is_finished, encode, available_actions


class AIagent_RL:
    def __init__(self, restore=False):
        self.action_value = dict()
        if not restore:
            self.init_value()
        else:
            self.restore()

    def init_value(self):
        state_list = itertools.product([0, 1, 2], repeat=9)

        for state in state_list:
            state = list(state)
            encoded_state = encode(state)
            available = available_actions(state)

            for action in available:
                encoded = encoded_state + str(action)
                self.action_value[encoded] = 0

    def policy(self, state, turn, epsilon=0.08):
        maxvalue = -99999
        minvalue = 99999
        encoded_state = encode(state)
        available = available_actions(state)
        action_list = []

        if np.random.rand(1) < epsilon:
            action_list = available
        else:
            if turn == 1:
                for action in available:
                    encoded = encoded_state + str(action)
                    if self.action_value[encoded] > maxvalue:
                        action_list = []
                        maxvalue = self.action_value[encoded]
                        action_list.append(action)
                    elif self.action_value[encoded] == maxvalue:
                        action_list.append(action)
            else:
                for action in available:
                    encoded = encoded_state + str(action)
                    if self.action_value[encoded] < minvalue:
                        action_list = []
                        minvalue = self.action_value[encoded]
                        action_list.append(action)
                    elif self.action_value[encoded] == minvalue:
                        action_list.append(action)

        return random.choice(action_list)

    def save(self):
        shutil.copy2("./data/save.dat", "./data/save_backup.dat")
        with open("./data/save.dat", 'w') as f:
            for key, value in self.action_value.items():
                f.write(key + ' ' + str(value) + '\n')
        print("saved!")

    def restore(self):
        with open("./data/save.dat", 'r') as f:
            for line in f:
                tmp = line.split()
                key = tmp[0]
                value = float(tmp[1])
                self.action_value[key] = value
        print("restored!")


class AIagent_Base:
    def policy(self, state, turn, epsilon=0):
        available = available_actions(state)
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
    def policy(self, state, turn, epsilon=0):
        available = available_actions(state)

        while True:
            ret = int(input("input [0 1 2 / 3 4 5 / 6 7 8] : "))
            if ret in available:
                break
        return ret
