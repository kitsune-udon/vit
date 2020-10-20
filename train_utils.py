from argparse import ArgumentParser
from warnings import warn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything


def validate_args(args):
    distributed = args.num_nodes > 1 or args.distributed_backend is not None
    if args.seed is None and distributed:
        warn("In a distributed running, '--seed' option must be specified.")


def main(dm_cls, model_cls, model_args, logger_name):
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--logger_name", type=str,
                        default=logger_name, help="logger name to identify")
    parser.add_argument("--save_top_k", type=int, default=1,
                        help="num of best models to save")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = dm_cls.add_argparse_args(parser)
    parser = model_cls.add_argparse_args(parser)
    args = parser.parse_args()
    print(args)

    validate_args(args)
    seed_everything(args.seed)

    dm = dm_cls.from_argparse_args(args)

    extracted_model_args = model_cls.extract_kwargs_from_argparse_args(
        args, **model_args)
    model = model_cls(**extracted_model_args)

    logger = TensorBoardLogger('tb_logs', name=args.logger_name)
    logger.log_hyperparams(args)
    logger.log_hyperparams(model_args)

    checkpoint = ModelCheckpoint(
        monitor='val_loss', filepath=None, save_top_k=args.save_top_k)

    trainer = pl.Trainer.from_argparse_args(args,
                                            deterministic=True,
                                            checkpoint_callback=checkpoint,
                                            logger=logger
                                            )

    trainer.fit(model, dm)
