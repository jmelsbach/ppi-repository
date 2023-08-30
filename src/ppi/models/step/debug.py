from ppi.data.datamodule import BaseDataModule
from ppi.models.step.step import StepModel
from lightning.pytorch import Trainer
from argparse import ArgumentParser

if __name__ == "__main__":
    # you might need to change the data path
    datamodule = BaseDataModule(
        data_path="data/madan/", batch_size=8, base_model="Rostlab/prot_bert_bfd"
    )

    parser = ArgumentParser()
    StepModel.add_model_specific_args(parser)
    args = parser.parse_args(args=[])

    model = StepModel(args)
    trainer = Trainer()
    trainer.fit(model, datamodule)
