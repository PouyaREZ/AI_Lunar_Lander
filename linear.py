'''###########################################
CS221 Final Project: Linear Q-Learning Implementation
Authors:
Kongphop Wongpattananukul (kongw@stanford.edu)
Pouya Rezazadeh Kalehbasti (pouyar@stanford.edu)
Dong Hee Song (dhsong@stanford.edu)
###########################################'''

import sys, math
import numpy as np
from collections import defaultdict
import random

import gym

import json

############################################################
class QLearningAlgorithm():
    def __init__(self, actions, discount, featureExtractor, weights, explorationProb=0.2, exploreProbDecay=0.99, explorationProbMin=0.01):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.exploreProbDecay = exploreProbDecay
        self.explorationProbMin = explorationProbMin
        self.weights = weights
        self.numIters = 0


    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions)
        else:
            return max((self.getQ(state, action), action) for action in self.actions)[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)
        # return 0.001

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
        # calculate gradient
        eta = self.getStepSize()
        if newState is not None:
            # find maximum Q value from next state
            V_opt = max(self.getQ(newState, possibleAction) for possibleAction in self.actions)
        else:
            # V_opt of end state is 0
            V_opt = 0.0
        Q_opt = self.getQ(state, action)
        target = reward + self.discount * V_opt
        # update weight
        for f, v in self.featureExtractor(state, action):
            self.weights[f] -=  eta * (Q_opt - target) * v
        # END_YOUR_CODE

# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
# maxVals = {'x':0, 'y':0, 'vx':0, 'vy':0, 'th':0, 'vth':0}
def modFeatureExtractor(state, action):
    # Action: 0: Nop, 1: fire left engine, 2: main engine, 3: right engine
    x, y, Vx, Vy, Th, VTh, LeftC, RightC = state
    x = np.round(x, decimals=1)
    y = np.round(y, decimals=1)
    Vx = np.round(Vx, decimals=1)
    Vy = np.round(Vy, decimals=1)
    Th = np.round(Th, decimals=1)
    VTh = np.round(VTh, decimals=1)
    
    # Feature config 0
    # features.append(((('Vx', Vx), action),1))
    # features.append(((('Vy', Vy), action),1))
    # features.append((((x, y, Vx, Vy), action),1))
    # features.append(((('VTh', VTh), action),1))
    
    
    # Feature config 1
    features = []
    features.append(((('x', x, Vx), action),1))
    features.append(((('y', y, Vy), action),1))
    features.append(((('Th', Th, VTh), action),1))
    features.append(((('C', LeftC, RightC), action),1))

    # Test config
    # features.append((('r', np.round(np.sqrt(x**2+y**2), decimals=1), action),1))
    # features.append((('Vr', np.round(np.sqrt(Vx**2+Vy**2), decimals=1), action),1))
    # features.append((('Th', np.round(Th*VTh, decimals=1), action),1))
    # features.append(((('LC', LeftC), action),1))
    # features.append(((('RC', RightC), action),1))

    # Test config 2
    # features.append((('x',  np.round(Vx/x, decimals=1), action),1))
    # features.append((('y', np.round(Vy/y, decimals=1), action),1))
    # features.append((('Th', np.round(VTh/Th, decimals=1), action),1))
    # features.append((('LC', LeftC, action),1))
    # features.append((('RC', RightC, action),1))
    
    ## Feature config 2: Separate features: SUCKED COMPARED TO COMBINED FEATURES!
    # features.append((('x', x, action),1))
    # features.append((('y', y, action),1))
    # features.append((('Vx', Vx, action),1))
    # features.append((('Vy', Vy, action),1))
    # features.append((('Th', Th, action),1))
    # features.append((('VTh', VTh, action),1))
    # features.append((('LC', RightC, action),1))
    # features.append((('RC', LeftC, action),1))
    
    return features

# Perform |numTrials| of the following:
# On each trial, take the MDP |mdp| and an RLAlgorithm |rl| and simulates the
# RL algorithm according to the dynamics of the MDP.
# Each trial will run for at most |maxIterations|.
# Return the list of rewards that we get for each trial.
def simulate(env, rl, numTrials=100, train=False, verbose=False, trialDemoInterval=10):
    totalRewards = []  # The rewards we get on each trial
    for trial in range(numTrials):
        state = env.reset()
        totalReward = 0
        iteration = 0
        while iteration < 3000:#iteration < 1000:
            action = rl.getAction(state)
            newState, reward, done, info = env.step(action)

            if train:
                rl.incorporateFeedback(state, action, reward, newState)
            totalReward += reward
            rl.explorationProb = max(rl.exploreProbDecay * rl.explorationProb,
                                         rl.explorationProbMin)
            state = newState
            iteration += 1

            if done:
                break

        totalRewards.append(totalReward)
        if verbose and trial % 20 == 0:
            print(('\n---- Trial {} ----'.format(trial)))
            print(('Mean(last 10 total rewards): {}'.format(np.mean(totalRewards[-100:]))))
            print(('Size(weight vector): {}'.format(len(rl.weights))))
        
    return totalRewards, rl.weights


# Helper functions for storing and loading the weights
import pickle
def saveF(obj, name):
    with open('weights/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def loadF(name):
    with open('weights/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


## Main variables
numFeatures = 4
numActions = 4
numEpochs = 1
numTrials = 2000 #30000
numTestTrials = 1000
trialDemoInterval = numTrials/2
discountFactor = 0.99
explorProbInit = 1.0
exploreProbDecay = 0.996
explorationProbMin = 0.01


# Main function
if __name__ == '__main__':
    # Initiate weights
    # np.random.seed(1)
    # random.seed(1)
    # Cold start weights
    weights = defaultdict(float)
    # Warm start weights
    # weights = loadF('weights')
    
    # TRAIN
    # for i in range(numEpochs):
    rl = QLearningAlgorithm([0, 1, 2, 3], discountFactor, modFeatureExtractor,
                            weights, explorProbInit, exploreProbDecay,
                            explorationProbMin)
    env = gym.make('LunarLander-v2')
    # env.seed(0)
    print('\n++++++++++++ TRAINING +++++++++++++')
    totalRewards, weights = simulate(env, rl, numTrials=numTrials, train=True, verbose=True, trialDemoInterval=trialDemoInterval)
    env.close()
    print('Average Total Training Reward: {}'.format(np.mean(totalRewards)))
    
    # Save Weights
    saveF(weights, 'weights')
    with open('weights.json', 'w') as fileOpen:
        json.dump(str(weights), fileOpen)
    
    
    # TEST
    print('\n\n++++++++++++++ TESTING +++++++++++++++')
    rl = QLearningAlgorithm([0, 1, 2, 3], discountFactor, modFeatureExtractor,
                            weights, 0.0, 1,
                            0.00)
    totalRewards, _ = simulate(env, rl, numTrials=numTestTrials, train=False, verbose=True, trialDemoInterval=trialDemoInterval)
    print('Average Total Testing Reward: {}'.format(np.mean(totalRewards)))

