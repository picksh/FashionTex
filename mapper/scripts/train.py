
import pytorch_lightning as pl
from argparse import ArgumentParser
import os,sys
sys.path.append('..')
from pytorch_lightning import loggers as pl_loggers
from mapper.training.coach import Coach
from mapper.datasets.latents_dataset import FashiondataModule
from mapper.options.train_options import TrainOptions

def train(args):

    dataloader_module=FashiondataModule(args)

    if args.checkpoint_path!=None and args.resume_training:
        coach=Coach.load_from_checkpoint(args.checkpoint_path,opts=args,strict=False)
    else:
        coach=Coach(args)

    tb_logger = pl_loggers.TensorBoardLogger(args.output_dir,name=args.exp_name)
    checkpoint_callback=pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="test_loss",
        mode="min",
        save_last=True,
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    trainer = pl.Trainer(
        gpus=-1,
        resume_from_checkpoint=args.checkpoint_path if args.resume_training else None,
        accelerator='ddp',
        fast_dev_run=False,
        logger=tb_logger,
        max_steps=args.max_steps,
        callbacks=callbacks,
        log_every_n_steps=1,
        flush_logs_every_n_steps=10,
        check_val_every_n_epoch=1,
        )
   
    if args.test:
        trainer.test(coach, datamodule=dataloader_module)
    else:
        trainer.fit(coach, datamodule=dataloader_module)

if __name__ == '__main__':
    opts = TrainOptions().parse()
    train(opts)
