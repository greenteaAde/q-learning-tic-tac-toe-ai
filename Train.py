import copy
from Tictactoe_Env import tictactoe
from Agent import AIagent_RL, AIagent_Base
from Functions import encode, available_actions

learning_rate = 0.4
total_episode = 50000000
train_episode = 500
verify_episode = 100

env = tictactoe()
agent = AIagent_RL()
agent_base = AIagent_Base()


def update(agent, state, next_state, learning_rate=0.4):
    state = encode(state)
    next_state = encode(next_state)
    agent.value[state] = agent.value[state] + learning_rate * (agent.value[next_state] - agent.value[state])


def train():
    episode = 0
    while episode < total_episode:
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
            # print("Episode: %d Winner : %d" % (episode, winner))

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
                    action = agent.policy(state, turn, available_actions(state))
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
        print("[Episode %d] Win : %d Draw : %d Lose : %d Win_rate: %.2f" % (episode, win, draw, lose, (win + draw) /
                                                                            verify_episode))
        agent.save()


if __name__ == "__main__":
    train()