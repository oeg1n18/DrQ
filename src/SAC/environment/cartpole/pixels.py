import collections

import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
from dm_control import suite
from dm_control.suite.wrappers.pixels import Wrapper


class gym_wrapper:
    def __init__(self, env):
        super(gym_wrapper, self).__init__()
        self.env = env

    def step(self, action):
        traj = self.env.step(action)
        obs = traj.observation['pixels']
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        h, w = obs.shape
        ch, cw = int(h / 2), int(w / 2)
        dh, dw = int(0.6 * h / 2), int(0.7 * w / 2)
        obs = obs[ch - dh:ch + dh, cw - dw:cw + dw]
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs, traj.reward, traj.last(), None

    def reset(self):
        traj = self.env.reset()
        obs = cv2.cvtColor(traj.observation['pixels'], cv2.COLOR_BGR2GRAY)
        h, w = obs.shape
        ch, cw = int(h/2), int(w/2)
        dh, dw = int(0.6*h/2), int(0.7*w/2)
        obs = obs[ch-dh:ch+dh, cw-dw:cw+dw]
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs

    def sample_action(self):
        return (np.random.rand() * 2) - 1


class ActionRepeat:
    def __init__(self, env, repeat):
        self.env = env
        self.repeat = repeat

    def sample_action(self):
        return (np.random.rand() * 2) - 1

    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            if done:
                break
        return obs, t_reward, done, info

    def reset(self):
        obs = self.env.reset()
        return obs


class StackFrames:
    def __init__(self, env, stack):
        self.env = env
        self.stack = collections.deque(maxlen=stack)

    def sample_action(self):
        return (np.random.rand() * 2) - 1

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.stack.append(obs)
        return np.array(self.stack).reshape(self.stack.maxlen, 84, 84), reward, done, info

    def reset(self):
        self.stack.clear()
        obs = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(obs)
        return np.array(self.stack).reshape(self.stack.maxlen, 84, 84)


def make_env(repeat=8, stack=8):
    env = Wrapper(suite.load(domain_name="cartpole", task_name="balance"), pixels_only=True)
    env = gym_wrapper(env)
    env = ActionRepeat(env, repeat)
    env = StackFrames(env, stack)
    return env


def run_env():
    env = make_env()
    obs = env.reset()
    done = False
    while not done:
        action = env.sample_action()
        obs_, reward, done, _ = env.step(action)
        plt.imshow(obs_[0], cmap='gray')
        plt.show()
        obs = obs_
        print(obs.shape, reward, obs_.shape, done)


