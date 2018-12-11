I
import tensorflow as tf
import numpy as np
import random
from collections import deque
import  pickle

OBSERVE = 500000.  # timesteps to observe before training



class DeepRLAgent:
    def __init__(self, checkpoint= None, action_space = None, observation = None, discount=0.95, exploration_rate=1.0):
        # init basic variables
        self.checkpoint = checkpoint
        self.action_space = action_space
        self.actions = action_space.n
        self.observation = observation
        self.discount = discount
        self.exploration_rate = exploration_rate
        self.epsilon = self.exploration_rate
        self.replayMemory = deque()
        self.timeStep = 0
        self.actionInput = tf.placeholder("float", [None, self.actions])
        self.yInput = tf.placeholder("float", [None])



        self.stateInput, self.QValue, self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2 = self.createNeuroNetwork()
        self.stateInputT, self.QValueT, self.W_conv1T, self.b_conv1T, self.W_conv2T, self.b_conv2T, self.W_conv3T, self.b_conv3T, self.W_fc1T, self.b_fc1T, self.W_fc2T, self.b_fc2T = self.createNeuroNetwork()

        self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1), self.b_conv1T.assign(self.b_conv1),
                                            self.W_conv2T.assign(self.W_conv2), self.b_conv2T.assign(self.b_conv2),
                                            self.W_conv3T.assign(self.W_conv3), self.b_conv3T.assign(self.b_conv3),
                                            self.W_fc1T.assign(self.W_fc1), self.b_fc1T.assign(self.b_fc1),
                                            self.W_fc2T.assign(self.W_fc2), self.b_fc2T.assign(self.b_fc2)]
        self.loss = tf.reduce_mean(
            tf.square(self.yInput - tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices=1)))
        self.trainStep = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(self.loss)
        self.loadInit()



    def createNeuroNetwork(self):
        # network weights
        W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev=0.01))
        b_conv1 = tf.Variable(tf.constant(0.01, shape=[32]))
        W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01))
        b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]))
        W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.01))
        b_conv3 = tf.Variable(tf.constant(0.01, shape=[64]))
        W_fc1 = tf.Variable(tf.truncated_normal([3136,512], stddev=0.01))
        b_fc1 = tf.Variable(tf.constant(0.01, shape=[512]))
        W_fc2 = tf.Variable(tf.truncated_normal([512,self.actions], stddev=0.01))
        b_fc2 = tf.Variable(tf.constant(0.01, shape=[self.actions]))
        stateInput = tf.placeholder("float", [None, 84, 84, 4])
        activition = tf.nn.relu
        h_conv1 = activition(tf.nn.conv2d(stateInput, W_conv1, [1, 4, 4, 1],"VALID") + b_conv1)
        h_conv2 = activition(tf.nn.conv2d(h_conv1, W_conv2, [1,2,2,1],"VALID") + b_conv2)
        h_conv3 = tf.reshape(activition(tf.nn.conv2d(h_conv2, W_conv3, [1,1,1,1],"VALID") + b_conv3),[-1, 3136])
        h_fc1 = activition(tf.matmul(h_conv3, W_fc1) + b_fc1)
        QValue = tf.matmul(h_fc1, W_fc2) + b_fc2
        return stateInput, QValue, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2






    def train(self, nextObservation, action, reward, done):
        newState = np.append(nextObservation, self.currentState[:, :, 1:], axis=2)
        self.replayMemory.append((self.currentState, action, reward, newState, done))
        if len(self.replayMemory) > 40000:  #replaymemory size
            self.replayMemory.popleft()
            minibatch = random.sample(self.replayMemory, 64)  # batchsize = 64
            stateBatch = [data[0] for data in minibatch]
            actionBatch = [data[1] for data in minibatch]
            rewardBatch = [data[2] for data in minibatch]
            nextStateBatch = [data[3] for data in minibatch]
            ybatch = []
            QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT: nextStateBatch})
            for i in range(0, 64):  # batchsize =64
                end = minibatch[i][4]
                if end:
                    ybatch.append(rewardBatch[i])
                else:
                    ybatch.append(rewardBatch[i] + self.discount * np.max(QValue_batch[i]))
            self.trainStep.run(feed_dict={
                self.yInput: ybatch,
                self.actionInput: actionBatch,
                self.stateInput: stateBatch
            })
            # tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(self.loss).run(feed_dict={
            #     self.yInput: ybatch,
            #     self.actionInput: actionBatch,
            #     self.stateInput: stateBatch
            # })
        if self.timeStep % 10000 == 0:# save session every 100000 timesteps
            self.save()
        self.currentState = newState
        self.timeStep += 1

    def act(self):
        QValue = self.QValue.eval(feed_dict={self.stateInput: [self.currentState]})[0]
        nextAction = np.zeros(self.actions)
        if random.random() <= self.epsilon:
            action = random.randrange(self.actions)
        else:
            action = np.argmax(QValue)
        nextAction[action] = 1
        if self.timeStep > OBSERVE:
            self.epsilonGreedy()
        return nextAction

    def epsilonGreedy(self):
        if self.epsilon > 0.1 :
            self.epsilon -= (self.exploration_rate - 0.1) / 100000

    def loadInit(self):
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        self.load()

    def save(self):
        # raise NotImplementedError('***Error: save to file  not implemented')
        # YOUR CODE HERE: save trained model to file
        self.saver.save(self.session, 'savedweights/network' + '-dqn', global_step=self.timeStep)
        self.session.run(self.copyTargetQNetworkOperation)
        pickle_out = open("replayMemory" , "wb")
        pickle.dump(self.replayMemory, pickle_out)
        pickle_out.close()
        print("replay memory and session saved")

    def load(self):
        if self.checkpoint and self.checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, self.checkpoint.model_checkpoint_path)
            print("Successfully loaded:", self.checkpoint.model_checkpoint_path)
            self.timeStep = int(self.checkpoint.model_checkpoint_path[25:])
            pickle_in = open("replayMemory", "rb")
            self.replayMemory = pickle.load(pickle_in)
        else:
            print("Could not find old network weights, start to train the model from scratch")

    def reset(self):
        # raise NotImplementedError('***Error: load from file not implemented')
        # YOUR CODE HERE: load trained model from file
        self.session.close()





