from dataloader import ObsDataset
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch as T



dataset = ObsDataset(data_dir="data/valid/", step_limit=5000)
dataloader = DataLoader(dataset, batch_size=32)


idx = np.random.randint(5000)
obs = dataset.__getitem__(idx)
print(obs.shape)
fig, ax = plt.subplots(2,2)
ax[0][0].imshow(obs[0], cmap='gray')
ax[1][0].imshow(obs[1], cmap='gray')
ax[0][1].imshow(obs[2], cmap='gray')
ax[1][1].imshow(obs[3], cmap='gray')
plt.tight_layout()
plt.show()


