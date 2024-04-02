import torch
import os
from copy import deepcopy
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from accelerate.utils import set_seed
from accelerate.logging import get_logger


class ParamDataset(Dataset):
    def __init__(self, args, augmentation=True):
        super().__init__()
        self.dataset_dir = args.dataset_dir
        self.model_pth_list = os.listdir(self.dataset_dir) #[:20]
        self.augmentation = augmentation
        self.augmentation_scale = args.augmentation_scale

    def __len__(self):
        return len(self.model_pth_list)

    def __getitem__(self, index: int):
        model_pth = os.path.join(self.dataset_dir, self.model_pth_list[index])
        weight_dict = torch.load(model_pth, map_location="cpu")

        weight = torch.Tensor([])
        for key in weight_dict:
            value = weight_dict[key]
            weight = torch.cat([weight, value.flatten()], dim=0)

        # augmentation: add noise
        if self.augmentation:
            noise = torch.randn_like(weight) * self.augmentation_scale
            weight = weight + noise

        data = {"weight_value": weight}
        return data

    def get_data_dim(self):
        if hasattr(self, 'data_dim'):
            return self.data_dim
        self.data_dim = len(self[0]['weight_value'])
        return self.data_dim


class ParamDataset2(Dataset):
    def __init__(self, args, augmentation=True):
        super().__init__()
        self.dataset_dir = args.dataset_dir
        self.data = torch.load(os.path.join(self.dataset_dir, "data.pt"), map_location="cpu")
        self.model_params = self.data["pdata"]
        self.augmentation = augmentation
        self.augmentation_scale = args.augmentation_scale

    def __len__(self):
        return self.model_params.shape[0]

    def __getitem__(self, index: int):
        weight = self.model_params[index]

        # # augmentation: add noise
        # if self.augmentation:
        #     noise = torch.randn_like(weight) * self.augmentation_scale
        #     weight = weight + noise

        data = {"weight_value": weight}
        return data

    def get_data_dim(self):
        if hasattr(self, 'data_dim'):
            return self.data_dim
        self.data_dim = self.model_params.shape[1]
        return self.data_dim

    def get_model(self):
        return self.data["model"]

    def get_layer_names(self):
        return self.data["train_layer"]
