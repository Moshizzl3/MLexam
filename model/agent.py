import gym
import random
import numpy as np
import tensorflow as tf
from keras.models import load_model
from statistics import mean, median
import time
import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import load_model
import numpy as np
from keras.optimizers import Adam
from pathlib import Path


class Agent():
    def __init__(self, warmup_games, max_memory, dnn_netowrk):
        self.score = 0
        self.dnn_network = dnn_netowrk
        self.warmup_games = warmup_games
        self.max_memory = max_memory
        self.model = None

    def getAction(self, state, game_number):
        if game_number > self.warmup_games:
            try:
                prediction = self.model.predict(np.array([state])).tolist()
                index_of_guess = prediction[0].index(max(prediction[0]))
                if (index_of_guess == 0):
                    action = 0
                    output = [1, 0, 0]
                elif (index_of_guess == 1):
                    action = 1
                    output = [0, 1, 0]
                elif (index_of_guess == 2):
                    action = 2
                    output = [0, 0, 1]
                return [action, output]
            except:
                action = random.randrange(0, 3)
                if (action == 0):
                    action = 0
                    output = [1, 0, 0]
                elif (action == 1):
                    action = 1
                    output = [0, 1, 0]
                elif (action == 2):
                    action = 2
                    output = [0, 0, 1]
                return [action, output]

        else:
            action = random.randrange(0, 3)
            if (action == 0):
                action = 0
                output = [1, 0, 0]
            elif (action == 1):
                action = 1
                output = [0, 1, 0]
            elif (action == 2):
                action = 2
                output = [0, 0, 1]

            return [action, output]

    def trainAgent(self, new_data):
        new_list = np.array(new_data)
        log_dir = "./data/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)

        try:
            training_data = np.load(
                'saved.npy', allow_pickle=True)
            print('Number of training_data: ' + str(len(training_data)))
            new_list = np.concatenate((training_data, new_list), axis=0)
            forget = 0 if self.max_memory > len(
                new_list) else len(new_list)-self.max_memory
            np.save('saved.npy', new_list)

            input_data = new_list[forget:, 0].tolist()
            output_data = new_list[forget:, 1].tolist()
            self.dnn_network.fit(
                input_data, output_data, verbose=1, epochs=10, callbacks=[tensorboard_callback])
            self.dnn_network.save('gamemodel.h5')
            self.model = load_model('gamemodel.h5')
        except:
            print(new_list)
            input_data = new_list[:, 0].tolist()
            output_data = new_list[:, 1].tolist()
            np.save('saved.npy', new_list)
            self.dnn_network.fit(
                input_data, output_data, verbose=1, epochs=10, callbacks=[tensorboard_callback])
            self.dnn_network.save('gamemodel.h5')
            self.model = load_model('gamemodel.h5')


class DnnNetwork():

    def __call__(self, *args, output_size):
        model = Sequential()  # creates new, empty model
        for arg in args:
            model.add(Dense(arg, activation='relu'))
            model.add(Dropout(0.2))

        model.add(Dense(output_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(
            lr=0.0001), metrics=['accuracy'])
        return model
