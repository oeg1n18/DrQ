import torch as T
from pytorch_lightning.loggers import TensorBoardLogger
from SAC.deepmind_pendulum_swingup.AutoEncoder.dataloader import ObsDataModule
from SAC.deepmind_pendulum_swingup.AutoEncoder.AutoEncoder import AutoencoderModule
import pytorch_lightning as pl


EPOCHS = 500
FRAME_SIZE = (84, 84)
LATENT_DIM = 50
FRAME_STACK = 8


if __name__ == "__main__":
    T.set_float32_matmul_precision('medium')
    dm = ObsDataModule(batch_size=128)
    model = AutoencoderModule.load_from_checkpoint("models/epoch=34-step=6860.ckpt", frame_size=FRAME_SIZE, latent_dim=LATENT_DIM, frame_stack=FRAME_STACK)
    logger = TensorBoardLogger("tb_logs", name="convAE")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="models", save_top_k=1, monitor="loss/train_loss")
    trainer = pl.Trainer(accelerator="gpu", max_epochs=EPOCHS, logger=logger,
                         default_root_dir="models/", enable_checkpointing=True,
                         callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm)
    trainer.validate(datamodule=dm)
