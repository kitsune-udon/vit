import torch.nn as nn

from imagenet_datamodule import ImageNetDataModule
from train_utils import main
from vit import VisionTransformer


class ImageNetVisionTransformer(VisionTransformer):
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
        n_classes = 1000
        self.classifier = nn.Linear(dim, n_classes)


vit_args = {
    "n_channels": 3,
    "patch_size": 32,
    "dim": 1024,
    "n_patches": (224//32)**2,
    "n_layers": 6,
    "n_heads": 8,
    "dropout": 0.,
}


if __name__ == '__main__':
    main(ImageNetDataModule,
         ImageNetVisionTransformer,
         vit_args,
         "vit_imagenet")
