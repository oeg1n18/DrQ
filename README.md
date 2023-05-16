# DrQ

---

This is a repository to test the reproduction of the 2021 ICLR paper [Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels](https://arxiv.org/abs/2004.13649).

The Regularized deep reinforcement learning method coined DrQ is tested with both continuous and discrete environments with SAC and DQN algorithms. The environments used are the deepmind control suite and atari gym with the results shown below. Read our **report** featured in the repo to find out more.


### Soft Actor Critic Results

---

![Alt text](./figures/SAC.png)


### DQN Results 

---

![combine_images](https://user-images.githubusercontent.com/41129056/236246866-f104a3ff-ced4-479d-8dbc-f28c7666f655.jpg)




## Using the Repository 

---

To recreate the results and run the code you will require a Nvidia Graphics Card with cuda-tookit>=11.

### SAC

---

**To install the required packages.** 

```
conda create --name env_name
conda activate env_name 

conda install python==3.9
pip install -r requirements.txt

conda activate env_name
```

The SAC repository has been developed in pycharm. A free community version can be downloaded from [here](https://www.jetbrains.com/pycharm/download/#section=linux)
To run the different algorithms open the project. SAC on state can be run from src/SAC/state/train_agent.py, SAC+AE on pixels can be run from src/SAC/image/train_agent.py
and DrQ regularized SAC+AE can be frun from src/SAC/drq_22/train_agent.py

**To view SAC Training Logs type the following in the terminal**
```bash
# AE Training 
tesnorboard --logdir src/SAC/autoencoder/tb_logs

# SAC on state 
tensorboard --logdir src/SAC/state/runs

# SAC+AE from Pixels
tensorboard --logdir src/SAC/image/runs

# DrQ Regularized SAC+AE from pixels
tensorboard --logdir src/SAC/drq_22/runs
```
### DQN

---

To run the DQN version algorithms type the command shown below in the terminal. To alter dqn implementations set the variable mode='Baseline' in the main.py file to run 
the standard NDDDQN. Alternatively set mode='DrQ' to run the DrQ regularized version of NDDDQN.
```python
python src/dqn/main.py
```



