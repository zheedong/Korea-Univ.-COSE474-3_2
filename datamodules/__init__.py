from collections.abc import Callable

from datamodules.datamodule import DataModule
import torchvision.transforms as transforms

def build_datamodule(cfg, **kwargs):
    return DataModule(cfg, **kwargs)


def build_transform(cfg, split: str) -> Callable:
    return transforms.Compose([
        # Add transforms here
        transforms.Resize(size=256),
        transforms.CenterCrop(size=256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


