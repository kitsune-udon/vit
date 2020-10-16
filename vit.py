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
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.cls_patch = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x):
        def extract_patches(x):
            bs = x.size()[0]
            p = self.patch_size
            x = rearrange(x, 'b c (h p2) (w p1) -> b (h w) (p1 p2 c)',
                          p1=p, p2=p)
            x = torch.cat((self.cls_patch.extend(bs, -1, -1), x), dim=1)
            return x

        z0 = self.position_embedding + self.patch_embedding(extract_patches(x))
        z = self.transformer(z0)
        z_mlp_head = F.layer_norm(z[:, 0])
        y = self.mlp(z_mlp_head)
        return y

    def calc_loss(self, x, label):
        y = F.log_softmax(self(x))
        loss = F.nll_loss(y, label)
        return loss

    def training_step(self, batch, batch_idx):
        x, label = batch
        loss = self.calc_loss(x, label)

        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        x, label = batch

        with torch.no_grad():
            y = F.log_softmax(self(x))
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
        parser.add_argument('--learning_rate', type=float, default=0.2)
        parser.add_argument('--weight_decay', type=float, default=1e-6)

        return parser
