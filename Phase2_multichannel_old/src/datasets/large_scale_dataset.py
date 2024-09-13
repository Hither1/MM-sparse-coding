from glob import glob
from .base_dataset import CsvDataset
import io
from PIL import Image


class LargeScaleDataset(CsvDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        if split == "test":
            split = "val"

        if split == "train":
            input_filename = f"Train_GCC-training_output.csv"
        elif split == "val":
            input_filename = f"Train_GCC-training_output.csv" #large_scale_benchmark.csv #Train_GCC-training_output.csv"

        img_key = "filepath"
        caption_key = "title"
        img_id_key = "image_id"

        super().__init__(
            *args,
            **kwargs,
            input_filename=input_filename,
            img_key=img_key,
            caption_key=caption_key,
            img_id_key=img_id_key,
            dataset_name="CC12M",
        )

    def __getitem__(self, index):
        return self.get_suite(index)