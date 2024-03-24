import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
import logging
import os
from typing import Optional
import random
import wandb

from train_utils import Engine


def main(args):
    formatter = logging.Formatter(
            fmt="%(asctime)s - %(filename)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    output_dir = "../outputs/task_training/{}/{}".format(args.name, args.exp_name)
    output_log = "../outputs/task_training/{}/{}/log.txt".format(args.name, args.exp_name)
    args.output_dir = output_dir
    if os.path.exists(output_log):
        output_log = "../outputs/task_training/{}/{}/log_seed{}_{}.txt".format(args.name, args.exp_name, args.seed, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    fh = logging.FileHandler(output_log)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info(OmegaConf.to_yaml(args))
    
    engine = Engine(args)
    engine.train()

    fh.close()
    logger.removeHandler(fh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_pth', type=str, default='../configs/task/mnist.yaml')
    conf_args = parser.parse_args()
    
    args = OmegaConf.load(conf_args.config_pth)
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    args.dataset_config.seed = seed
    main(args)
    
    