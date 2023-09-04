from argparse import ArgumentParser

from lightning.pytorch import Trainer

from ppi.data.datamodule import BaseDataModule
from ppi.models.baseline.siamese import SiameseBaseLineModel

if __name__ == "__main__":
    # you might need to change the data path
    datamodule = BaseDataModule(
        data_path="data/madan/", batch_size=8, base_model="facebook/esm2_t6_8M_UR50D"
    )

    model = SiameseBaseLineModel()
    trainer = Trainer(precision='16-mixed', limit_train_batches=0.1, limit_val_batches=0.2, limit_test_batches=0.3)
    trainer.fit(model, datamodule)
