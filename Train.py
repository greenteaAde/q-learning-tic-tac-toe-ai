import copy
import numpy as np
import random
from Tictactoe_Env import tictactoe
from Agent import AIagent_RL, AIagent_Base
from Functions import encode, available_actions

learning_rate = 0.4
discount_factor = 0.9
epsilon = 0.08
# epsilon_list = [0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
train_episode1 = 350
train_episode2 = 150
verify_episode = 100
total_episode = (train_episode1 + train_episode2) * 100000

env = tictactoe()
agent = AIagent_RL(restore=False)
agent_base = AIagent_Base()


def greedy_policy(agent, state, turn):
    maxvalue = -99999
    minvalue = 99999
    available = available_actions(state)
    encoded_state = encode(state)
    action_list = []

    if turn == 1:
        for action in available:
            encoded = encoded_state + str(action)
            if agent.action_value[encoded] > maxvalue:
                action_list = []
                maxvalue = agent.action_value[encoded]
                action_list.append(action)
            elif agent.action_value[encoded] == maxvalue:
                action_list.append(action)
    else:
        for action in available:
            encoded = encoded_state + str(action)
            if agent.action_value[encoded] < minvalue:
                action_list = []
                minvalue = agent.action_value[encoded]
                action_list.append(action)
            elif agent.action_value[encoded] == minvalue:
                action_list.append(action)

    return random.choice(action_list)


def update(agent, state, action, reward, next_state, next_turn, done, learning_rate=0.4, discount_factor=0.9):
    encoded = encode(state) + str(action)
    if not done:
        next_action = greedy_policy(agent, next_state, next_turn)
        next_encoded = encode(next_state) + str(next_action)
        agent.action_value[encoded] = agent.action_value[encoded] + learning_rate \
                                      * (reward + discount_factor * agent.action_value[next_encoded]
                                         - agent.action_value[encoded])
    else:
        agent.action_value[encoded] = agent.action_value[encoded] \
                                      + learning_rate * (reward - agent.action_value[encoded])


def train():
    win_rate_list = []
    win_rate_mean = []

    episode = 0
    while True:  # episode < total_episode:

        # epsilon = random.choice(epsilon_list)

        # training stage1 (self-training)
        for _ in range(train_episode1):
            episode += 1
            done = 0
            env.reset()
            state = copy.copy(env.state)

            while not done:
                turn = copy.copy(env.turn)
                action = agent.policy(state, turn, epsilon=epsilon)
                next_state, done, reward, winner = env.step(action)
                update(agent, state, action, reward, next_state, turn % 2 + 1, done,
                       learning_rate=learning_rate, discount_factor=discount_factor)
                state = copy.copy(next_state)

        # training stage2 (vs agent_base)
        for i in range(train_episode2):
            episode += 1
            done = 0
            env.reset()
            state = copy.copy(env.state)

            j = 0
            while not done:
                j += 1
                turn = copy.copy(env.turn)
                if (i + j) % 2 == 1:
                    action = agent.policy(state, turn, epsilon=epsilon)
                else:
                    action = agent_base.policy(state, turn, available_actions(state))
                next_state, done, reward, winner = env.step(action)
                if done:
                    update(agent, state, action, reward, next_state, turn % 2 + 1, done,
                           learning_rate=learning_rate, discount_factor=discount_factor)
                state = copy.copy(next_state)

        # verification stage
        win = lose = draw = 0
        for i in range(verify_episode):
            done = 0
            env.reset()
            state = copy.copy(env.state)

            j = 0
            while not done:
                j += 1
                turn = copy.copy(env.turn)
                if (i + j) % 2 == 1:
                    # epsilon 0
                    action = agent.policy(state, turn, epsilon=0)
                else:
                    action = agent_base.policy(state, turn, epsilon=0)
                next_state, done, reward, winner = env.step(action)
                state = copy.copy(next_state)

            if winner == 0:
                draw += 1
            elif (i + j) % 2 == 1:
                win += 1
            else:
                lose += 1
        win_rate = (win + draw) / verify_episode
        print("[Episode %d] Win : %d Draw : %d Lose : %d Win_rate: %.2f" % (episode, win, draw, lose, win_rate))
        agent.save()

        # if win_rate == 1.0: break

        # print status (each train_episode * 100)
        win_rate_list.append(win_rate)
        if episode % ((train_episode1 + train_episode2) * 100) == 0:
            mean = np.mean(win_rate_list)
            win_rate_mean.append(np.round(mean, 2))
            win_rate_list.clear()
            print("[ ", end='')
            for x in win_rate_mean:
                print("%.2f" % x, end=' ')
            print("]")


if __name__ == "__main__":
    train()
