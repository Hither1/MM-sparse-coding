from .f30k_caption_karpathy_datamodule import F30KCaptionKarpathyDataModule
from .coco_caption_karpathy_datamodule import CocoCaptionKarpathyDataModule
from .conceptual_caption_datamodule import ConceptualCaptionDataModule
from .sbu_caption_datamodule import SBUDataModule
from .large_scale_datamodule import LargeScaleDataModule

_datamodules = {
    "f30k": F30KCaptionKarpathyDataModule,
    "coco": CocoCaptionKarpathyDataModule,
    "gcc": ConceptualCaptionDataModule,
    "sbu": SBUDataModule,
    "largescale": LargeScaleDataModule,
}
