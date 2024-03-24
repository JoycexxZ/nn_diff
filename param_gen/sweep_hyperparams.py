import wandb
import argparse
import torch
import random
import numpy as np
from omegaconf import OmegaConf
import yaml

from task_training import main


config_pth = '../configs/task/mnist.yaml'
args = OmegaConf.load(config_pth)

with open("../configs/task/mnist_search_param.yaml", "r") as f:
    sweep_config = yaml.load(f, Loader=yaml.FullLoader)
project_name = sweep_config["name"]
args.name = project_name

run = wandb.init(project=project_name, config=sweep_config)

args.dataset_config.train_bs = wandb.config.batch_size
args.lr = wandb.config.learning_rate
args.seed = wandb.config.seed
args.exp_name = "bs{}_lr{}".format(
    args.dataset_config.train_bs, args.lr)
run.name = args.exp_name

seed = args.seed
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
args.dataset_config.seed = seed

main(args)
