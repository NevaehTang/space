import cv2
import gym
import os
import sys
from DeepRLAgentnew3 import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

env = gym.make('SpaceInvaders-v0')
# env = gym.make('Breakout-ram-v0')
env.reset()

def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110,:]
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(84,84,1))





def train():
    checkpoint = tf.train.get_checkpoint_state("savedweights")
    nextObservation, reward, done, info = env.step(0)
    agent = DeepRLAgent(checkpoint=checkpoint, observation=nextObservation, action_space= env.action_space, discount=0.95, exploration_rate= 1.0)
    agent.currentState = np.stack((preprocess(nextObservation), preprocess(nextObservation), preprocess(nextObservation), preprocess(nextObservation)), axis=2)
    agent.currentState = np.squeeze(agent.currentState)
    rewards = 0
    while True:
        action = agent.act()
        nextObservation, reward, done, info = env.step(np.argmax(np.array(action)))
        rewards+=reward
        if done:
            print(agent.timeStep, rewards)
            rewards =0
            nextObservation = env.reset()
        agent.train(preprocess(nextObservation), action, reward, done)


env.close()

def test():
    checkpoint = tf.train.get_checkpoint_state("savedweights")
    nextObservation, reward, done, info = env.step(0)
    agent = DeepRLAgent(checkpoint=checkpoint, observation=nextObservation, action_space=env.action_space,
                        discount=0.95, exploration_rate=1.0)
    totalRewards = 0
    for i in range(10):
        agent.currentState = np.stack((preprocess(nextObservation), preprocess(nextObservation), preprocess(nextObservation), preprocess(nextObservation)), axis=2)
        agent.currentState = np.squeeze(agent.currentState)
        rewards = 0
        while True:
            action = agent.act()
            nextObservation, reward, done, info = env.step(np.argmax(np.array(action)))
            rewards+=reward
            if done:
                print(rewards)
                totalRewards+=rewards
                env.reset()
                nextObservation, reward, done, info = env.step(0)
                # agent.reset()
                break
    print("Finished 10 games with average reward ", totalRewards/10)





program_name = sys.argv[0]
command = sys.argv[1]

if command == "-train":
    train()
elif command == "-test":
    test()
else:
    print ("Please use command line: -train / -test")



