from SAC.deepmind_pendulum_swingup.environment.cartpole_swingup.pixels import make_env
import torch as T
from buffer import ReplayBuffer

STEPS_LIMIT = 5000
device = "cuda" if T.cuda.is_available() else "cpu"

env = make_env()

'''
just need to collect test and valid
'''
data_store = ReplayBuffer(STEPS_LIMIT, (8, 84, 84), 1, device)
step_count = 0
episode_count = 0
while step_count < STEPS_LIMIT:
    obs = env.reset()
    done = False

    episode_count += 1
    while not done:
        step_count += 1
        if step_count == STEPS_LIMIT:
            done = True
        action = env.sample_action()
        obs_, reward, done, info = env.step(action)
        data_store.put((obs, action, reward, obs_, int(done)))
        obs = obs_
    print("avg episode steps: ", step_count/episode_count, "total steps: ", step_count)

print("buffer length: ", len(data_store.buffer))
data_store.save("data/valid")
