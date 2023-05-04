import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

WINDOW=10
ALPHA=0.2
LENGTH=400
TITLE_SIZE=21
LABEL_SIZE=18

fig, ax = plt.subplots(1, 3, figsize=(25, 8))

image_return = pd.read_csv("returns/image_return.csv").to_numpy()[:LENGTH, :]
state_return = pd.read_csv("returns/state_return.csv").to_numpy()[:LENGTH, :]
drq22_return = pd.read_csv("returns/drq22_return.csv").to_numpy()[:LENGTH, :]
Q_var = pd.read_csv("returns/Q_var.csv").to_numpy()[:LENGTH, :]


def smooth_data(scores, window=50):
    series = pd.Series(scores[:, 2], index=scores[:, 1])
    smoothed = series.rolling(window=window).mean()
    smoothed_std = series.rolling(window=window).std()
    return smoothed, smoothed_std

def plot(ax, steps, scores, std, name, alpha=0.2):
    ax.plot(steps, scores, label=name)
    ax.fill_between(steps, scores-std, scores+std, alpha=alpha)
    return ax

"""
### Step Times are ###
state: 0.00766
Image: 0.169
drq22: 0.291
drq11: 0.178
drq12: 0.271
drq21: 0.196
"""
labels = ["state", "image", "drq22", "drq12", "drq21", "drq11"]
data = np.array([0.00766, 0.169, 0.291, 0.271, 0.196, 0.178])
ax[2].bar(labels, data*125)
ax[2].set_title("Runtime Performance", fontsize=TITLE_SIZE)
ax[2].set_ylabel("Average Episode Time (s)", fontsize=LABEL_SIZE)
ax[2].set_xlabel("(c) SAC Implementation", fontsize=LABEL_SIZE)
ax[2].xaxis.set_tick_params(labelsize=17, rotation=0)


smoothed_Qvar, Qvar_std = smooth_data(Q_var, window=1)
series = pd.Series(Q_var[:, 2], index=Q_var[:, 1])
smoothed = series.rolling(window=30).mean()
max = series.rolling(window=3).max()
max[max>35] = 35.
min = series.rolling(window=3).min()
ax[1].plot(Q_var[:, 1], smoothed, label="Q-value mse")
ax[1].fill_between(Q_var[:, 1], min, max, alpha=0.3)
ax[1].set_xlabel("(b) Training Step", fontsize=LABEL_SIZE)
ax[1].set_ylabel("mse", fontsize=LABEL_SIZE)
ax[1].set_title("Q-value Image Augmentation MSE", fontsize=TITLE_SIZE)
ax[1].legend()


image_smooth, image_std = smooth_data(image_return, window=WINDOW)
state_smooth, state_std = smooth_data(state_return, window=WINDOW)
drq22_smooth, drq22_std = smooth_data(drq22_return, window=WINDOW)

ax[0] = plot(ax[0], image_return[:, 1], image_smooth, image_std, "SAC+AE:  Pixels", alpha=ALPHA)
ax[0] = plot(ax[0], state_return[:, 1], state_smooth, state_std, "SAC: State", alpha=ALPHA)
ax[0] = plot(ax[0], drq22_return[:, 1], drq22_smooth, drq22_std, "DrQ K=2 M=2: Pixels", alpha=ALPHA)
ax[0].set_xlabel("(a) Environment Step", fontsize=LABEL_SIZE)
ax[0].set_ylabel("Episode Return", fontsize=LABEL_SIZE)
ax[0].set_title("Episode Returns", fontsize=TITLE_SIZE)
ax[0].legend(fontsize="13")
plt.show()