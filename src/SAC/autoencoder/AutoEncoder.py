import torch as T
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=50):
        super().__init__()

        self.conv1 = nn.Conv2d(input_dim[0], 32, 3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=0)

        self.fc1 = nn.Linear(39200, latent_dim)
        self.layernorm = nn.LayerNorm(50)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

    def conv(self, x, detach=False):
        batch_size = x.shape[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(batch_size, -1)
        if detach:
            return x.detach()
        else:
            return x

    def forward(self, x, detach_encoder=False):
        x = self.conv(x, detach=detach_encoder)
        x = self.fc1(x)
        x = self.layernorm(x)
        x = T.tanh(x)
        return x



class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim=50):
        super().__init__()

        self.deconv1 = nn.ConvTranspose2d(32, 32, 3, stride=1, padding=0)
        self.deconv2 = nn.ConvTranspose2d(32, 32, 3, stride=1, padding=0)
        self.deconv3 = nn.ConvTranspose2d(32, 32, 3, stride=1, padding=0)
        self.deconv4 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=0, output_padding=1)

        self.fc1 = nn.Linear(latent_dim, 39200)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.ConvTranspose2d):
            nn.init.orthogonal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, z):
        batch_size = z.shape[0]
        z = self.fc1(z)
        z = z.view(batch_size, 32, 35, 35)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.deconv3(z))
        z = F.relu(self.deconv4(z))
        return z



def run_ae(shape=(32, 1, 84, 84)):
    img = T.rand(shape)
    encoder = Encoder((1, 84, 84))
    decoder = Decoder((1, 84, 84))
    out = encoder(img)
    out = decoder(out)
    assert img.shape == out.shape
    return True


class AutoencoderModule(pl.LightningModule):
    def __init__(self, frame_size=(84,84), latent_dim=50, frame_stack=8):
        super().__init__()
        self.encoder = Encoder((1,) + frame_size, latent_dim=latent_dim)
        self.decoder = Decoder((1,) + frame_size, latent_dim=latent_dim)
        self.loss_fn = nn.MSELoss()
        self.fs = frame_stack

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x = batch.reshape(batch.shape[0] * self.fs, batch.shape[2], batch.shape[3])
        z = self.encoder(x[:, None, :, :])
        x_hat = self.decoder(z)
        x_hat = x_hat.squeeze().reshape(batch.shape[0], self.fs, batch.shape[2], batch.shape[3])
        loss = self.loss_fn(x_hat, batch)
        self.log("loss/train_loss", loss.clone().detach().cpu())
        return loss.mean()

    def validation_step(self, batch, batch_idx):
        x = batch.reshape(batch.shape[0] * self.fs, batch.shape[2], batch.shape[3])
        z = self.encoder(x[:, None, :, :])
        x_hat = self.decoder(z)
        x_hat = x_hat.squeeze().reshape(batch.shape[0], self.fs, batch.shape[2], batch.shape[3])
        loss = self.loss_fn(x_hat, batch)
        self.log("loss/val_loss", loss.clone().detach().cpu())
        return loss.mean()

    def test_step(self, batch, batch_idx):
        x = batch.reshape(batch.shape[0] * self.fs, batch.shape[2], batch.shape[3])
        z = self.encoder(x[:, None, :, :])
        x_hat = self.decoder(z)
        x_hat = x_hat.squeeze().reshape(batch.shape[0], self.fs, batch.shape[2], batch.shape[3])
        loss = self.loss_fn(x_hat, batch)
        self.log("loss/test_loss", loss.clone().detach().cpu())
        return loss.mean()

    def configure_optimizers(self):
        optimizer = T.optim.SGD(self.parameters(), lr=0.0001)
        return optimizer

    def encode(self, x, detach_encoder=False):
        if len(x.shape) == 3:
            batch_size = 1
            x = x[None, :]
        else:
            batch_size = x.shape[0]
        x = x.view(x.shape[0] * self.fs, 84, 84)
        codes = self.encoder(x[:, None, :, :], detach_encoder=detach_encoder)
        codes = codes.squeeze().reshape(x.shape[0], -1)
        code = codes.reshape(batch_size, -1)
        return code

    def decode(self, z):
        batch_size = z.shape[0]
        z = z.reshape(z.shape[0] * self.fs, -1)
        x_hat = self.decoder(z)
        return x_hat.reshape(batch_size, self.fs, x_hat.shape[2], x_hat.shape[3])