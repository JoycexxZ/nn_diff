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

with open("../configs/task/mnist_gen.yaml", "r") as f:
    sweep_config = yaml.load(f, Loader=yaml.FullLoader)
project_name = sweep_config["name"]

run = wandb.init(project=project_name, config=sweep_config)

args.name = project_name
args.seed = wandb.config.seed
run.name = "seed_{}".format(args.seed)
args.exp_name = project_name

seed = args.seed
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
args.dataset_config.seed = seed

main(args)
