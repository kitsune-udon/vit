import torch.nn as nn

from cifar10_datamodule import CIFAR10DataModule
from train_utils import main
from vit import VisionTransformer


class CIFAR10VisionTransformer(VisionTransformer):
    def __init__(self, *args,
                 dim=None,
                 patch_size=None,
                 n_patches=None,
                 n_layers=None,
                 n_heads=None,
                 dropout=None,
                 **kwargs):
        super().__init__(*args,
                         dim=dim,
                         patch_size=patch_size,
                         n_patches=n_patches,
                         n_layers=n_layers,
                         n_heads=n_heads,
                         dropout=dropout,
                         **kwargs)
        n_classes = 10
        self.classifier = nn.Linear(dim, n_classes)


vit_args = {
    "n_channels": 3,
    "patch_size": 8,
    "dim": 128,
    "n_patches": (32//8)**2,
    "n_layers": 4,
    "n_heads": 4,
    "dropout": 0.,
}


if __name__ == '__main__':
    main(CIFAR10DataModule, CIFAR10VisionTransformer, vit_args, "vit_cifar10")
