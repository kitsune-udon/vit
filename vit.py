import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pl_bolts.optimizers import LARSWrapper, LinearWarmupCosineAnnealingLR
from torch.optim import SGD

from argparse_utils import from_argparse_args


class MSA(nn.Module):  # Multiheaded self-attention
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.scaler = dim ** -0.5

        self.u_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.u_msa = nn.Linear(dim, dim, bias=False)

    def forward(self, z):
        qkv = self.u_qkv(z)
        q, k, v = rearrange(qkv,
                            'b n (qkv h d) -> qkv b h n d',
                            qkv=3,
                            h=self.n_heads)

        prod = torch.einsum('bhid,bhjd->bhij', q, k) * self.scaler
        a = prod.softmax(dim=-1)
        r = torch.einsum('bhij,bhjd->bhid', a, v)
        r = rearrange(r, 'b h n d -> b n (h d)')
        r = self.u_msa(r)

        return r


class Block(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )
        self.msa = MSA(dim, n_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, z):
        z = self.msa(self.norm1(z)) + z
        z = self.mlp(self.norm2(z)) + z

        return z


class Transformer(nn.Module):
    def __init__(self, n_layers, dim, n_heads, dropout):
        super().__init__()
        t = nn.ModuleList()

        for _ in range(n_layers):
            t.append(Block(dim, n_heads, dropout))

        self.layers = t

    def forward(self, x):
        t = x

        for layer in self.layers:
            t = layer(t)

        return t


class VisionTransformer(pl.LightningModule):
    def __init__(self,
                 *args,
                 n_channels=None,
                 dim=None,
                 patch_size=None,
                 n_patches=None,
                 n_layers=None,
                 n_heads=None,
                 dropout=None,
                 learning_rate=None,
                 weight_decay=None,
                 warmup_epochs=None,
                 max_epochs=None,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.transformer = Transformer(n_layers, dim, n_heads, dropout)
        self.cls_patch = nn.Parameter(torch.randn(1, 1, dim))
        self.position_embedding = nn.Parameter(
            torch.randn(1, n_patches + 1, dim))
        self.patch_embedding = nn.Linear(patch_size*patch_size*n_channels, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        def extract_patches(x):
            ps = self.hparams.patch_size
            x = rearrange(x, 'b c (h ph) (w pw) -> b (h w) (ph pw c)',
                          ph=ps, pw=ps)
            return x

        def append_cls(x):
            bs = x.size()[0]  # batch size
            c = self.cls_patch.expand(bs, -1, -1)
            x = torch.cat((c, x), dim=1)
            return x

        x = extract_patches(x)
        x = self.patch_embedding(x)
        x = append_cls(x)
        z0 = self.position_embedding + x
        z = self.transformer(z0)
        mlp_head = self.norm(z[:, 0])
        y = self.classifier(mlp_head)

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
            loss = F.nll_loss(y, label, reduction='sum')
            pred = torch.argmax(y, dim=1)
            c = pred.eq(label).sum().item()

        return {'loss': loss, 'correct': c, 'batch_size': x.size()[0]}

    def validation_epoch_end(self, outputs):
        def _sum(xs):
            if isinstance(xs[0], list):
                return sum(map(sum, xs))
            elif isinstance(xs[0], torch.Tensor):
                return sum(map(torch.sum, xs))
            else:
                return sum(xs)

        total = _sum([out['batch_size'] for out in outputs])

        loss_sum = _sum([out['loss'] for out in outputs])
        val_loss = loss_sum / total

        correct = _sum([out['correct'] for out in outputs])
        val_acc = correct / total * 100

        logs = {'val_loss': val_loss, 'val_acc': val_acc}
        results = {'log': logs}

        return results

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(),
                        lr=self.hparams.learning_rate,
                        momentum=0.9,
                        weight_decay=self.hparams.weight_decay)
        optimizer = LARSWrapper(optimizer)

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=0.01
        )

        return [optimizer], [scheduler]

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return from_argparse_args(cls, args, **kwargs)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--learning_rate', type=float, default=0.2)
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--warmup_epochs', type=int, default=10)

        return parser
