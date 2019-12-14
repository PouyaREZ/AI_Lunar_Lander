import sys, math
import numpy as np
from collections import deque
import random
import copy
import gym
import keras
from keras.models import Sequential
from keras.layers import Dense


############################################################
class QLearningAlgorithm():
    def __init__(self, actions, discount, weights, explorationProb=0.2, exploreProbDecay=0.99, explorationProbMin=0.01, batchSize=32):
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.exploreProbDecay = exploreProbDecay
        self.explorationProbMin = explorationProbMin
        self.weights = weights
        self.numIters = 0
        self.model = NeuralNetwork(batchSize, weights)
        self.cache = deque(maxlen=1000000)

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        if np.random.rand() < self.explorationProb:
            return random.choice(self.actions)
        else:
            predScores = self.model.predict(state)[0]
            return np.argmax(predScores)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, states, actions, rewards, newStates, dones):
        # initialize variable
        states = np.squeeze(states)
        newStates = np.squeeze(newStates)
        X = states
        y = self.model.predict(states)
        # calculate gradient
        targets = rewards + self.discount*(np.amax(self.model.predict(newStates), axis=1))*(1-dones)
        ind = np.array([i for i in range(len(states))])
        y[[ind], [actions]] = targets
        # update weight
        self.model.fit(X, y)

    def updateCache(self, state, action, reward, newState, done):
        self.cache.append((state, action, reward, newState, done))

# neural network
class NeuralNetwork():
    def __init__(self, batchSize = 32, weights=None):
        self.model = Sequential()
        self.model.add(Dense(100, input_dim=8, activation='relu'))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(4, activation='linear'))
        adam = keras.optimizers.adam(lr=0.001)
        self.model.compile(loss='mse', optimizer=adam)
        if isinstance(weights, str):
            self.model.load_weights(weights)

    def predict(self, state):
        return self.model.predict_on_batch(state)

    def fit(self, X, y):
        self.model.fit(X, y, epochs=1,  verbose=0)

    def save(self, weights):
        self.model.save_weights(weights)

# Perform |numTrials| of the following:
# On each trial, take the MDP |mdp| and an RLAlgorithm |rl| and simulates the
# RL algorithm according to the dynamics of the MDP.
# Each trial will run for at most |maxIterations|.
# Return the list of rewards that we get for each trial.
def simulate(env, rl, numTrials=10, train=False, verbose=False,
             trialDemoInterval=10, batchSize=32):
    totalRewards = []  # The rewards we get on each trial
    for trial in range(numTrials):
        state = np.reshape(env.reset(), (1,8))
        totalReward = 0
        iteration = 0
        while iteration <= 500:
        # while True:
            action = rl.getAction(state)
            newState, reward, done, info = env.step(action)
            newState = np.reshape(newState, (1,8))
            # Appending the new results to the deque
            rl.updateCache(state, action, reward, newState, done)

            # update
            totalReward += reward
            state = newState
            iteration += 1

            if verbose == True and trial % trialDemoInterval == 0:
                still_open = env.render()
                if still_open == False: break
            
            # Conducting memory replay
            if len(rl.cache) < batchSize: # Waiting till memory size is larger than batch size
                continue
            else:
                batch = random.sample(rl.cache, batchSize)
                states = np.array([sample[0] for sample in batch])
                actions = np.array([sample[1] for sample in batch])
                rewards = np.array([sample[2] for sample in batch])
                newStates = np.array([sample[3] for sample in batch])
                dones = np.array([sample[4] for sample in batch])

                if train:
                    rl.incorporateFeedback(states, actions, rewards, newStates,
                                           dones)
                    
                    rl.explorationProb = max(rl.exploreProbDecay * rl.explorationProb,
                                            rl.explorationProbMin)

            if done:
                break
        
        totalRewards.append(totalReward)
        if verbose:
            print(('Trial {} Total Reward: {}'.format(trial, totalReward)))
            print(('Mean(last 10 total rewards): {}'.format(np.mean(totalRewards[-10:]))))
        
    return totalRewards

## Main variables
# np.random.seed(0)
numEpochs = 300
numTrials = 1
numTestTrials = 1000
trialDemoInterval = numTrials/2
discountFactor = 0.99
explorProbInit = 1.0
exploreProbDecay = 0.999
explorationProbMin = 0.01
batchSize = 64

if __name__ == '__main__':
    # Initiate weights
    # Cold start weights
    # weights = None
    # Warm start weights
    weights = 'weights.h5'

    # TRAIN
    # print('\n++++++++++++ TRAINING +++++++++++++')
    # rl = QLearningAlgorithm([0, 1, 2, 3], discountFactor, weights,
    #                         explorProbInit, exploreProbDecay,
    #                         explorationProbMin, batchSize)
    # env = gym.make('LunarLander-v2')
    # # env.seed(0)

    # for i in range(numEpochs):
    #     totalRewards = simulate(env, rl, numTrials=numTrials, train=True, verbose=False,
    #                             trialDemoInterval=trialDemoInterval, batchSize=batchSize)
    #     print('Average Total Reward in Trial {}/{}: {}'.format(i, numEpochs, np.mean(totalRewards)))
    # env.close()
    # # Save Weights
    # rl.model.save('weights.h5')

    # TEST
    print('\n\n++++++++++++++ TESTING +++++++++++++++')
    env = gym.make('LunarLander-v2')
    env.seed(3)
    rl = QLearningAlgorithm([0, 1, 2, 3], discountFactor, weights, 0.0, 0.0, 0.0, batchSize)
    totalRewards = simulate(env, rl, numTrials=numTestTrials, train=False, verbose=False, trialDemoInterval=trialDemoInterval)
    env.close()
    print('Average Total Testing Reward: {}'.format(np.mean(totalRewards)))