import gym
import random
from agent import Agent
from agent import DnnNetwork
from matplotlib import pyplot as plt

time_step = 500
warmup_games = 200
games_amount = warmup_games + 50
score_requirement = -499
scores = []

env = gym.make("Acrobot-v1")


dnn_network = DnnNetwork()
agent = Agent(warmup_games, 10_000, dnn_network(
    32, 64, 32, output_size=3), "acrobot-data.npy", "acrobot-gamemodel.h5")
state = env.reset()  # return the initial state of the game
action = random.randrange(0, 3)  # first action is random
for x in range(games_amount):
    print("-------------- game number:", x)
    training_data = []
    agent.score = 0
    game_memory = []
    for _ in range(time_step):
        state, reward, done, info = env.step(action)
        if x < 1 or x > warmup_games:
            env.render()

        action, output = agent.get_action(state, x)  # get action from agent
        agent.score += reward
        if done:
            scores.append(agent.score)
            break
        game_memory.append([state, output])
    print(agent.score)

    # If agent score is higher than the requirements, game is saved to memory
    if agent.score >= score_requirement:
        for data in game_memory:  # loops through game memory and formats it correctly
            training_data.append([data[0].tolist(), data[1]])

        agent.trainAgent(training_data)
        # puts a lower limit to the score requirements
        score_requirement = -200 if score_requirement > -200 else score_requirement + 2
        print('Agent score:', agent.score)
        print('Score requirement:', score_requirement)

    # show development i scores
    if x > 0:
        plt.ion()  # makes the plot not block the flow
        plt.plot(scores)  # data to plot
        plt.xlim([len(scores)-100, len(scores)])  # adds limit to chart
        plt.draw()  # draws the data
        plt.pause(0.001)  # add small pause, so we can see the chart

    env.reset()

plt.show()
