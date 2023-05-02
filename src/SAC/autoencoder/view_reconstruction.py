
from src.SAC.autoencoder.AutoEncoder import AutoencoderModule
from src.SAC.autoencoder.dataloader import ObsDataset
import matplotlib.pyplot as plt
import numpy as np
import torch as T

model = AutoencoderModule.load_from_checkpoint("models/epoch=17-step=3528.ckpt",
                                     frame_size=(84, 84), latent_dim=50)


dataset = ObsDataset(data_dir="data/train/", step_limit=5000)


idx = np.random.randint(5000)
print(idx)
obs = dataset.__getitem__(idx)

z = model.encode(obs[None, :])
obs_ = model.decode(z)
obs_ = obs_.detach().cpu().numpy()[0]
obs = obs.numpy()

fig, ax = plt.subplots(2, 2, figsize=(10,10))

ax[0][0].imshow(obs[0], cmap='gray')
ax[0][0].set_title("Target Channel 0", fontsize=18)
ax[1][0].imshow(obs[1], cmap='gray')
ax[1][0].set_title("Target Channel 1", fontsize=18)



ax[0][1].imshow(obs_[0], cmap='gray')
ax[0][1].set_title("Reconstructed Channel 0", fontsize=18)
ax[1][1].imshow(obs_[1], cmap='gray')
ax[1][1].set_title("Reconstructed Channel 1", fontsize=18)



fig.suptitle("Autoencoder State Reconstructions", fontsize=25)
plt.tight_layout(h_pad=3)
plt.subplots_adjust(top=0.9)
plt.show()
