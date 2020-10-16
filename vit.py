import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.optim import AdamW

from argparse_utils import extract_kwargs_from_argparse_args


class VisionTransformer(pl.LightningModule):
    def __init__(self,
                 *args,
                 dim=None,
                 patch_size=None,
                 n_patches=None,
                 learning_rate=None,
                 weight_decay=None,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.cls_patch = nn.Parameter(torch.randn(1, 1, dim))
        self.position_embedding = nn.Parameter(
            torch.randn(1, n_patches + 1, dim))
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        def extract_patches(x):
            p = self.hparams.patch_size
            x = rearrange(x, 'b c (h p2) (w p1) -> b (h w) (p1 p2 c)',
                          p1=p, p2=p)
            return x

        def append_cls(x):
            bs = x.size()[0]
            c = self.cls_patch.expand(bs, -1, -1)
            x = torch.cat((c, x), dim=1)
            return x

        x = extract_patches(x)
        x = self.patch_embedding(x)
        x = append_cls(x)
        z0 = self.position_embedding + x
        z = self.transformer(z0)
        mlp_head = self.layer_norm(z[:, 0])
        y = self.mlp(mlp_head)
        return y

    def calc_loss(self, x, label):
        y = F.log_softmax(self(x), dim=1)
        loss = F.nll_loss(y, label)
        return loss

    def training_step(self, batch, batch_idx):
        x, label = batch
        loss = self.calc_loss(x, label)

        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        x, label = batch

        with torch.no_grad():
            y = F.log_softmax(self(x), dim=1)
            loss = F.nll_loss(y, label)

        return loss

    def validation_epoch_end(self, outputs):
        ls = [loss.mean() for loss in outputs]
        val_loss = sum(ls) / len(ls)

        logs = {'val_loss': val_loss}
        results = {'log': logs}

        return results

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          lr=self.hparams.learning_rate,
                          weight_decay=self.hparams.weight_decay)

        return optimizer

    @classmethod
    def extract_kwargs_from_argparse_args(cls, args, **kwargs):
        return extract_kwargs_from_argparse_args(cls, args, **kwargs)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=1e-2)

        return parser
