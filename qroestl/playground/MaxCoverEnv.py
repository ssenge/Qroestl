import numpy as np
import gym
from gym import spaces

from qroestl.utils import Utils


class MaxCoverEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, p):
        super(MaxCoverEnv, self).__init__()
        self.p = p
        n_actions = p.nS
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, p.nS))#, dtype=np.float32)

    def reset(self):
        self.steps = 0
        #self.total_reward = 0
        self.state = np.array([0] * self.p.nS)
        return np.array(self.state)

    covers = lambda self: len(set(Utils.flatten2d([self.p.C[i] for i, s in enumerate(self.state) if s == 1])))

    def step(self, action):
        self.steps += 1
        hit = self.state[action]
        self.state[action] = 1
        done = np.sum(self.state > 0) == self.p.k

        #reward = self.covers() - (1 if self.steps>self.p.nS else 0) #(self.steps/10)# if done else 0
        #if done:
            #reward = self.covers() + (1 if self.steps==self.p.nS else 0)
            #reward = 10*self.covers() + 5*(self.p.nS - min(self.p.nS, self.steps))
        #    reward = 2*self.covers() - np.clip(self.steps, self.p.nS, 2*self.p.nS) # 5*(self.p.nS - min(self.p.nS, self.steps))
        #else:
        #    reward = self.covers()
        if done:
            #reward = 2*self.covers() - self.steps
            reward = -0.5*self.steps+3*self.covers()#1*self.covers() #+ (125-(self.steps))
                #self.total_reward+5#max(0, 125-(self.steps*self.steps))/20#0.5#self.covers()#3#-0.01*self.steps
        else:
            reward = -hit#0 if hit == 1 else 1
                #-4*hit#0#-0.1*self.steps#self.covers()/10 if self.steps < self.p.nS else 0
            #self.total_reward-=hit
        info = {}
        return self.state, reward, done, info

    def render(self, mode='console'):
        print(f'Sets: {self.state}')

    def close(self):
        pass