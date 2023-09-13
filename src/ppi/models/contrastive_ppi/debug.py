# Contrastive PPI
from ppi.data.datamodule import ContrastiveDataModule
from ppi.models.contrastive_ppi.contrastive_ppi import ContrastivePPI
from lightning.pytorch import Trainer
import lightning.pytorch as pl

if __name__ == "__main__":
    datamodule = ContrastiveDataModule(data_path="data/madan", batch_size=256)
    model = ContrastivePPI(base_model="Rostlab/prot_bert")
    trainer = Trainer(precision="16-mixed")
    trainer.fit(model, datamodule)
