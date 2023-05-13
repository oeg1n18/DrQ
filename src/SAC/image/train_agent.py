import torch
from src.SAC.environment.cartpole.pixels import make_env
from agent import SAC_Agent
import yaml
import time
import os

with open('config.yaml', 'r') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)


if __name__ == '__main__':
    env = make_env()
    agent = SAC_Agent(cfg)
    step_count = 0
    episode_count = 0
    score_list = []
    step_list = []

    while step_count < cfg["steps_limit"]:
        start_time = time.time()
        obs = env.reset()
        score, done = 0.0, False
        episode_count += 1
        while not done:
            if step_count == cfg["steps_limit"]:
                done = True

            action, log_prob = agent.choose_action(torch.FloatTensor(obs))
            action = action.detach().cpu().numpy()
            obs_, reward, done, _ = env.step(action)

            agent.memory.put((obs, action, reward, obs_, done))

            score += reward

            obs = obs_

            if agent.memory.size() > agent.batch_size:
                agent.learn()

            step_count += 1
        end_time = time.time()
        total_time = end_time - start_time

        print(
            "Episode:{}, Step_Count:{} Avg_Score:{:.1f}, Episode_time:{:.2f}, Remaining_time: {:.1f}".format(episode_count,
                                                                                                             step_count,
                                                                                                             score,
                                                                                                             total_time,
                                                                                                             (cfg[
                                                                                                                  "steps_limit"] - step_count) * total_time / 125))

        agent.writer.add_scalar("episode_return", score, step_count)
        agent.writer.add_scalar("episode_time", total_time, episode_count)

