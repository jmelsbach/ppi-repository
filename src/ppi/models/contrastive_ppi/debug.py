# Contrastive PPI
from ppi.data.datamodule import ContrastiveDataModule
from ppi.models.contrastive_ppi.contrastive_ppi import ContrastivePPI
from lightning.pytorch import Trainer
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner


if __name__ == "__main__":
    base_model = 'Rostlab/prot_bert'
    datamodule = ContrastiveDataModule(
        data_path="data/madan",
        batch_size=64,
        base_model=base_model,
        max_length=1024,
    )
    model = ContrastivePPI(
        base_model=base_model,
        scheduler="cosine",
    )
    trainer = Trainer(
        precision="16-mixed",
        max_epochs=10,
    )
    # tuner = Tuner(trainer)
    # tuner.lr_find(model, datamodule)
    trainer.fit(model, datamodule)
