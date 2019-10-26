from collections import deque
import random

import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=8000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.tau = .125
        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        state_shape = self.env.observation_space.shape
        model = Sequential(layers=[
            Dense(24, input_dim=state_shape[0], activation='relu'),
            Dense(48, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.env.action_space.n, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, observation, done):
        self.memory.append([state, action, reward, observation, done])

    def replay(self):
        batch_size = 32
        if (len(self.memory) < batch_size):
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, observation, done = sample
            target = self.target_model.predict(state)
            if (done):
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(observation)[0])
                target[0][action] = reward + Q_future * self.gamma
                self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if (np.random.random() < self.epsilon):
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def save_model(self, file):
        self.model.save(file)
