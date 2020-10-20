import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from argparse_utils import extract_kwargs_from_argparse_args


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
        total = sum([out['batch_size'] for out in outputs])

        ls = [out['loss'] for out in outputs]
        val_loss = sum(ls) / total

        correct = sum([out['correct'] for out in outputs])
        val_acc = correct / total * 100

        logs = {'val_loss': val_loss, 'val_acc': val_acc}
        results = {'log': logs}

        return results

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          lr=self.hparams.learning_rate,
                          weight_decay=self.hparams.weight_decay)

        scheduler = StepLR(optimizer, step_size=1, gamma=1.)

        return [optimizer], [scheduler]

    @classmethod
    def extract_kwargs_from_argparse_args(cls, args, **kwargs):
        return extract_kwargs_from_argparse_args(cls, args, **kwargs)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=1e-2)

        return parser
