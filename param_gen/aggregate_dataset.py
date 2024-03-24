'''
generate the dataset for the neural network

'''
import torch
import argparse
import os


def aggregate_dataset(args):
    folder_dir = os.path.join(args.param_save_dir, args.folder_name)
    param_list = os.listdir(folder_dir)
    for param_pth in param_list:
        param = torch.load(os.path.join(folder_dir, param_pth))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_save_dir', type=str, default='/scratch/yufan/nn_diff')
    parser.add_argument("--folder_name", type=str, default="gen_param_2_gen_param_2")
    parser.add_argument("save_dir", type=str, default="/scratch/yufan/nn_diff/param_dataset/")
    args = parser.parse_args()
    aggregate_dataset(args)