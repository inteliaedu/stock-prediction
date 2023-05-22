import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers, initializers
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

def crear_learning_rate_ciclico(config, X_train):
    steps_per_epoch = int(X_train.shape[0] / int(config["training"]["batch_size"]))
    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=float(config["training"]["min_learning_rate"]),
    maximal_learning_rate=float(config["training"]["max_learning_rate"]),
    scale_fn=lambda x: 1/(2.**(x-1)),
    step_size=2 * steps_per_epoch)
    step = np.arange(0, int(config["training"]["epochs"]) * steps_per_epoch)
    lr = clr(step)
    plt.plot(step, lr)
    plt.xlabel("Steps")
    plt.ylabel("Learning Rate")
    plt.show()
    return clr

class MyModel(keras.Model):
    def __init__(self, config):
        super(MyModel, self).__init__()
        # Adding the first LSTM layer and some Dropout regularisation
        self.lstm1 = layers.LSTM(units=config["lstm_size"], return_sequences=True, kernel_initializer=initializers.GlorotUniform(seed=1))
        self.dropout1 = layers.Dropout(config["dropout"])

        # Adding a second LSTM layer and some Dropout regularisation
        self.lstm2 = layers.LSTM(units=config["lstm_size"], return_sequences=False, kernel_initializer=initializers.GlorotUniform(seed=2))
        self.dropout2 = layers.Dropout(config["dropout"])

        # Adding the output layer
        self.dense = layers.Dense(units=1, activation='linear', kernel_initializer=initializers.GlorotUniform(seed=3))

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.dropout1(x)

        x = self.lstm2(x)
        x = self.dropout2(x)

        outputs = self.dense(x)
        return outputs