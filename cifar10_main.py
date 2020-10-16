import torch.nn as nn

from cifar10_datamodule import CIFAR10DataModule
from train_utils import main
from vit import VisionTransformer


class CIFAR10VisionTransformer(VisionTransformer):
    def __init__(self, *args,
                 dim=None,
                 patch_size=None,
                 **kwargs):
        super().__init__(*args, dim=dim, patch_size=patch_size, **kwargs)
        self.transformer = nn.Identity()
        self.patch_embedding = nn.Linear(patch_size*patch_size*3, dim)
        n_classes = 10
        self.mlp = nn.Linear(dim, n_classes)


vit_args = {
    "patch_size": 16,
    "dim": 32,
    "n_patches": (32*32)//(16*16),
}


if __name__ == '__main__':
    main(CIFAR10DataModule, CIFAR10VisionTransformer, vit_args, "vit_cifar10")
