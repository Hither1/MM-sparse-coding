from ..datasets import LargeScaleDataset
from .datamodule_base import BaseDataModule


class LargeScaleDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return LargeScaleDataset

    @property
    def dataset_name(self):
        return "largescale"