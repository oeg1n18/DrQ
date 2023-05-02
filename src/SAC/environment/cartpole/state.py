import collections

from dm_control import suite
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt




class gym_wrapper:
    def __init__(self, env):
        super(gym_wrapper, self).__init__()
        self.env = env

    def step(self, action):
        traj = self.env.step(action)
        obs = np.concatenate((traj.observation['position'], traj.observation['velocity']))
        return obs, traj.reward, traj.last(), None

    def reset(self):
        traj = self.env.reset()
        obs = np.concatenate((traj.observation['position'], traj.observation['velocity']))
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


def make_env(repeat=8):
    env = suite.load(domain_name="cartpole", task_name="balance")
    env = gym_wrapper(env)
    env = ActionRepeat(env, repeat)
    return env


def run_env():
    env = make_env()
    obs = env.reset()
    done = False
    step_count = 0
    while not done:
        step_count += 1
        action = env.sample_action()
        obs_, reward, done, _ = env.step(action)
        print(obs.shape, reward, action, obs_.shape, done)
        obs = obs_
    print("Episode Length: ", step_count)