import numpy as np
import argparse
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from scDLM import data, models


if __name__ == "__main__":
    # Parse arguments add all hyperparameters to the parser
    parser = argparse.ArgumentParser(description="Model Hyperparameters")
    # parser.add_argument(
    #     "--model_type",
    #     type=str,
    #     default="Separate_Multihead_Residual_CNN",
    #     help="Model type",
    # )
    parser.add_argument("--out_folder", help="the output folder")

    parser.add_argument(
        "--conv_layers",
        help="the number of the large convolutional layers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--conv_repeat",
        help="the number of the convolutional conv block",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--kernel_number", help="the number of the kernels", default=512
    )
    parser.add_argument("--kernel_length", help="the length of the kernels", default=15)
    parser.add_argument(
        "--filter_number", help="the number of the filters", default=256
    )
    parser.add_argument("--kernel_size", help="the size of the kernels", default=3)