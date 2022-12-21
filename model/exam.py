import numpy as np
from keras.models import load_model

#Show data
trainingData = np.load('saved.npy', allow_pickle=True)
print(trainingData[0])

#desribe a model
model = load_model('gamemodel.h5')  # loads pre-trained model
print(model.summary())