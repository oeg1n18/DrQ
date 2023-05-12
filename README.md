# DrQ

---

This is a repository to test the reproduction of the 2021 ICLR paper [Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels](https://arxiv.org/abs/2004.13649).

The Regularized deep reinforcement learning method is tested with both continous and discrete environments with SAC and DQN algorithms. The environments used are the deepmind control suite and atari gym. 
The results of the SAC and DQN results can be seen below. Read our **report** to find out more.


## Soft Actor Critic Results

---

![Alt text](./figures/SAC.png)


### DQN Results 

---

![combine_images](https://user-images.githubusercontent.com/41129056/236246866-f104a3ff-ced4-479d-8dbc-f28c7666f655.jpg)




To recreate the results and run the code you will require a Nvidia Graphics Card.

**To install the required packages.** 

```
conda create --name env_name
conda activate env_name 

conda install python==3.9
pip install -r requirements.txt
```

** To run the SAC Algorithms **
```python
# SAC Standard implmenetation on state observations
python src/SAC/state/train_agent.py

# SAC Standard implmenetation on Image Observations
python src/SAC/image/train_agent.py

# SAC with DrQ Regularization on Image Observations
python src/SAC/drq_22/train_agent.py
```
