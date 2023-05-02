import torch as T
from torchvision.transforms import transforms

import SAC.deepmind_pendulum_swingup.AutoEncoder.buffer as buffer
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

EPOCHS = 100
FRAME_SIZE = (84, 84)
LATENT_DIM = 50


class ObsDataset(Dataset):
    def __init__(self, data_dir="data", augmentations=None, step_limit=10000, input_dim=(8, 84, 84), action_dim=1):
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'
        self.ReplayBuffer = buffer.ReplayBuffer(step_limit, input_dim, action_dim, self.device)
        self.ReplayBuffer.load(data_dir)
        self.augmentations = augmentations

    def __len__(self):
        return len(self.ReplayBuffer.buffer)

    def __getitem__(self, idx):
        traj = self.ReplayBuffer.buffer[idx]
        obs = T.tensor(traj[0], dtype=T.float32)
        if self.augmentations:
            obs = self.augmentations(obs)
        return obs

class ObsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'
        self.test_step_limit = 5000
        self.valid_step_limit = 5000
        self.train_step_limit = 25000
        self.transform = transforms.RandomCrop(84, padding=4, padding_mode='edge')

    def setup(self, stage: str):
        self.data_test = ObsDataset(data_dir="data/test/", step_limit=self.test_step_limit,
                                    augmentations=self.transform)
        self.data_valid = ObsDataset(data_dir="data/valid/", step_limit=self.valid_step_limit,
                                     augmentations=self.transform)
        self.data_train = ObsDataset(data_dir="data/train/", step_limit=self.train_step_limit,
                                     augmentations=self.transform)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_valid, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=True)

    def predict_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=True)


def run_dataset():
    ds = ObsDataset(data_dir="data/test/", step_limit=5000, input_dim=(8,84,84))
    for obs in ds:
        print(obs.shape, obs.dtype, type(obs))
        break


run_dataset()