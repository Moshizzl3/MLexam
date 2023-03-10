import gym
import random
from agent import Agent
from agent import DnnNetwork
from matplotlib import pyplot as plt

time_step = 200
warmup_games = 5_000
games_amount = warmup_games + 200
score_requirement = -199
scores = []

env = gym.make("MountainCar-v0")


dnn_network = DnnNetwork()
agent = Agent(warmup_games, 10_000, dnn_network(
    64, 128, 64, output_size=3), "cart-data.npy", "cart-gamemodel.h5")
state = env.reset()  # return the initial state of the game
action = random.randrange(0, 3)  # first action is random
for x in range(games_amount):
    print("-------------- game number:", x)
    training_data = []
    agent.score = 0
    game_memory = []
    env.reset()
    for _ in range(time_step):
        if x < 2 or x > warmup_games:
            env.render()

        state, reward, done, info = env.step(action)
        action, output = agent.get_action(state, x)  # get action from agent
        reward = 1 if state[0] > -0.2 and x < warmup_games else reward
        agent.score += reward
        if done:
            scores.append(agent.score)
            break
        game_memory.append([state, output])
    print("Agent score:", agent.score)

    # If agent score is higher than the requirements, game is saved to memory
    if agent.score >= score_requirement:
        for data in game_memory:  # loops through game memory and formats it correctly
            training_data.append([data[0].tolist(), data[1]])

        agent.trainAgent(training_data)
        # puts a lower limit to the score requirements
        score_requirement = -150 if score_requirement > -150 else score_requirement + 1
        print('Agent score:', agent.score)
        print('Score requirement:', score_requirement)

    # show development in scores
    if x > 4800:
        plt.ion()  # makes the plot not block the flow
        plt.plot(scores)  # data to plot
        plt.xlim([len(scores)-100, len(scores)])  # adds limit to chart view
        plt.draw()  # draws the data
        plt.pause(0.001)  # add small pause, so we can see the chart

    env.reset()

plt.show()
