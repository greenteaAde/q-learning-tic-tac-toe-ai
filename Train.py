import copy
import numpy as np
from Tictactoe_Env import tictactoe
from Agent import AIagent_RL, AIagent_Base
from Functions import encode, available_actions

learning_rate = 0.4
train_episode = 500
verify_episode = 100
total_episode = train_episode * 100000

env = tictactoe()
agent = AIagent_RL(restore=True)
agent_base = AIagent_Base()


def update(agent, state, next_state, learning_rate=0.4):
    state = encode(state)
    next_state = encode(next_state)
    agent.value[state] = agent.value[state] + learning_rate * (agent.value[next_state] - agent.value[state])


def train():
    win_rate_list = []
    win_rate_mean = []

    episode = 0
    while episode < total_episode:
        # training stage
        for _ in range(train_episode):
            episode += 1
            done = 0
            env.reset()
            state = copy.copy(env.state)
            while not done:
                turn = copy.copy(env.turn)
                action = agent.policy(state, turn, available_actions(state))
                next_state, done, winner = env.step(action)
                update(agent, state, next_state, learning_rate=learning_rate)
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
                    action = agent.policy(state, turn, available_actions(state), epsilon=0)
                else:
                    action = agent_base.policy(state, turn, available_actions(state))
                next_state, done, winner = env.step(action)
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

        # print status (each train_episode * 100)
        win_rate_list.append(win_rate)
        if episode % (train_episode * 100) == 0:
            mean = np.mean(win_rate_list)
            win_rate_mean.append(round(mean,2))
            win_rate_list.clear()
            print("Recent Win Rates : {}".format(win_rate_mean))
            if mean > 80:
                break


if __name__ == "__main__":
    train()
