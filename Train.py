import matplotlib.pyplot as plt
import copy
from Tictactoe_Env import tictactoe
from Agent import AIagent_RL, AIagent_Base

learning_rate = 0.4
discount_factor = 0.9
epsilon = 0.08
train_episode = 500
verify_episode = 100

env = tictactoe()
agent = AIagent_RL(restore=False)
agent_base = AIagent_Base()

episode_plt = []
q_plt = []
wr_plt = []


def update(agent, state, action, reward, next_state, next_turn, done, learning_rate=0.4, discount_factor=0.9):
    value = agent.q(state, action)
    if not done:
        next_action = agent.policy(next_state, next_turn, epsilon=0)
        next_value = agent.q(next_state, next_action)
        agent.assign_q(state, action, value + learning_rate * (reward + discount_factor * next_value - value))
    else:
        agent.assign_q(state, action, value + learning_rate * (reward - value))


def train():
    episode = 0
    avg_q = 0.0
    while True:  # episode < total_episode:

        # training stage (self-training)
        for _ in range(train_episode):
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

        # verification stage
        winner = win = lose = draw = 0
        stage = 0
        for i in range(verify_episode):
            stage += 1
            done = 0
            env.reset()
            state = copy.copy(env.state)

            j = 0
            while not done:
                j += 1
                turn = copy.copy(env.turn)
                if (i + j) % 2 == 1:
                    action = agent.policy(state, turn, epsilon=0)
                    avg_q += agent.q(state, action)
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
        avg_q /= stage
        print("[Episode %d] Win : %d Draw : %d Lose : %d Win_rate: %.2f" % (episode, win, draw, lose, win_rate))

        agent.save()
        episode_plt.append(episode)
        q_plt.append(avg_q)
        wr_plt.append(win_rate)

        # plot data
        if episode == 100000:
            plt.plot(episode_plt, q_plt)
            plt.xlabel('Episode')
            plt.ylabel('Average Q-Value')
            plt.show()
            plt.plot(episode_plt, wr_plt)
            plt.xlabel('Episode')
            plt.ylabel('Win Rate [vs Base Agent]')
            plt.show()


if __name__ == "__main__":
    train()
