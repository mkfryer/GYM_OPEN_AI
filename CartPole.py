import gym
import random
import numpy as np 
from Memory import Memory
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os 

class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.model = Sequential()
        self.model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(self.action_size, activation='linear'))
        self.model.compile(loss='mse', optimizer = Adam(lr = self.learning_rate))

    def train(self, state, target_f, epochs):
        self.model.fit(state, target_f, epochs=epochs, verbose=0)

    def predict(self, state):
        return self.model.predict(state)

class CartPoleAgent():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        # discount rate
        self.gamma = 0.95  
        # exploration rate  
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.memory = Memory(2000)
        self.DQN = DQN(state_size, action_size)
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.DQN.predict(state)
        return np.argmax(act_values[0])  # returns action

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = self.memory.sample(batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * np.amax(self.DQN.predict(next_state)[0])
            target_f = self.DQN.predict(state)
            target_f[0][action] = target
            self.DQN.train(state, target_f, epochs=1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = CartPoleAgent(state_size, action_size)
    # memory_size = 2000
    for e in range(500):
        state = env.reset()
        state = np.reshape(state, [1, 4])
        for time_t in range(500):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("Total Reward:", reward)
                break
        agent.replay(32)
    env.close()

