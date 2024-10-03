import argparse
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import resnet
from dataset import get_cifar10

from mlx_catalyst.engine import Trainer 
# from mlx_catalyst.data import Dataset
from mlx_catalyst.loggers import WandBLogger

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument(
    "--arch",
    type=str,
    default="resnet20",
    choices=[f"resnet{d}" for d in [20, 32, 44, 56, 110, 1202]],
    help="model architecture",
)
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--cpu", action="store_true", help="use cpu only")

def eval_fn(model, inp, tgt):
    return mx.mean(mx.argmax(model(inp), axis=1) == tgt)

def main(args):
    mx.random.seed(args.seed)

    model = getattr(resnet, args.arch)()

    print("Number of params: {:0.04f} M".format(model.num_params() / 1e6))

    optimizer = optim.Adam(learning_rate=args.lr)
    train_data, test_data = get_cifar10(args.batch_size)
    # wnblogger = WandBLogger("resnet20-cifar-catalyst", "1")
    # wnblogger.init()

    trainer = Trainer(
        model,
        optimizer=optimizer,
        loss_fn=nn.losses.cross_entropy,
        train_loader=train_data,
        dataset_keys=["image", "label"],
        val_loader=test_data,
        epochs=args.epochs,
        # logger=wnblogger,
    )
    trainer.train()

if __name__ == "__main__":
    args = parser.parse_args()
    if args.cpu:
        mx.set_default_device(mx.cpu)
    main(args)
