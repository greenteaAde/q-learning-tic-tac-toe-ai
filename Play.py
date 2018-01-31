import copy
from Tictactoe_Env import tictactoe
from Agent import AIagent_RL, AIagent_Base, Human_agent
from Functions import available_actions, encode


def update(agent, state, next_state, learning_rate=0.4):
    state = encode(state)
    next_state = encode(next_state)
    agent.value[state] = agent.value[state] + learning_rate * (agent.value[next_state] - agent.value[state])


env = tictactoe()
agent1 = Human_agent()
agent2 = AIagent_RL(restore=True)


def play():
    done = 0
    env.reset()
    state = copy.copy(env.state)

    i = 0
    while not done:
        i += 1
        turn = copy.copy(env.turn)
        if i % 2 == 1:
            action = agent1.policy(state, turn, available_actions(state), epsilon=0)
        else:
            action = agent2.policy(state, turn, available_actions(state), epsilon=0)
        next_state, done, winner = env.step(action)
        update(agent2, state, next_state)
        state = copy.copy(next_state)
        env.render()

    if winner == 0:
        print("Draw!")
    else:
        print("Winner is agent %d!" % winner)
    agent2.save()

if __name__ == "__main__":
    play()
