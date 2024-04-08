'''
Core Training Part for param generation
TODO: save accuracy while save model weights
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools
from tqdm import tqdm
import wandb
import logging
from torch import autograd
import time
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
import os

from models.convnet import ConvNet2_mnist, ConvNet3_cifar


def build_model_and_opt(args):
    # model
    input_channels = 1 if args.dataset_config.dataset == 'mnist' else 3
    if args.model == 'ConvNet2' and args.dataset_config.dataset == 'mnist':
        model = ConvNet2_mnist(input_channels, init=args.init).cuda()
    elif args.model == 'ConvNet3' and args.dataset_config.dataset == 'cifar':
        model = ConvNet3_cifar(input_channels).cuda()
    else:
        raise ValueError('Unknown model: {}'.format(args.model))
    # optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                              momentum=args.momentum, 
                              weight_decay=args.weight_decay)
    else:
        raise ValueError('Unknown optimizer: {}'.format(args.optimizer))
    # scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    return model, optimizer, scheduler

def build_loader(args):
    # transforms
    input_size = 28 if args.dataset == 'mnist' else 32
    if args.dataset == 'mnist':
        normal_value = {'mean': [0.1307,], 'std': [0.3081,]}
    elif args.dataset == 'cifar':
        normal_value = {"mean": [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    test_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(normal_value["mean"], normal_value["std"])
                                ])
    if args.transforms:
        train_transform = transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(input_size, 4),
                                        transforms.ToTensor(), # first, convert image to PyTorch tensor
                                        transforms.Normalize(normal_value["mean"], normal_value["std"])
                                        ])
    else:
        train_transform = transforms.Compose([
                                        transforms.ToTensor(), # first, convert image to PyTorch tensor
                                        transforms.Normalize(normal_value["mean"], normal_value["std"])
                                        ])
    
    # datasets
    if args.dataset == 'mnist':
        train_set = MNIST(root=args.root, train=True, download=True, transform=train_transform)
        test_set = MNIST(root=args.root, train=False, download=True, transform=test_transform)
    elif args.dataset == 'cifar':
        train_set = CIFAR10(root=args.root, train=True, download=True, transform=train_transform)
        test_set = CIFAR10(root=args.root, train=False, download=True, transform=test_transform)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    # dataloaders
    train_loader = DataLoader(train_set, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.test_bs, shuffle=False, num_workers=args.num_workers)

    return train_loader, test_loader

def build_criterion(args):
    criterion = nn.CrossEntropyLoss()
    return criterion


class Engine:
    def __init__(self, args) -> None:
        self.args = args
        self.train_loader, self.test_loader = build_loader(args.dataset_config)
        self.model, self.optimizer, self.scheduler = build_model_and_opt(args)
        self.criterion = build_criterion(args)
        
        self.output_dir = args.output_dir
            
        wandb.init(project=args.name, name=args.exp_name)
        # wandb.config = {"batch_size": args.dataset_config,
        #                "lr": args.lr}
        
        self.logger = logging.getLogger()
        
    def train(self):
        self.model.train()
        for epoch in range(self.args.epochs):
            for images, labels in self.train_loader:
                images, labels = images.cuda(), labels.cuda()
                output = self.model(images)
                loss = self.criterion(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            
            if epoch % self.args.eval_per_epoch == 0 or epoch == self.args.epochs - 1:
                accuracy, loss = self.evaluate()
                
                wandb.log({"accuracy": accuracy, "loss": loss}, step=epoch)
                self.logger.info("test on epoch {}".format(epoch))
                self.logger.info("test accuracy: {:.4f}, test loss: {:.4f}".format(accuracy, loss))
            
            if self.args.param_save_dir != "" and (epoch+1) >= self.args.save_start_epoch:
                param_save_folder = "{}/{}_{}".format(self.args.param_save_dir, self.args.name, self.args.exp_name)
                if not os.path.exists(param_save_folder):
                    os.makedirs(param_save_folder)
                param_save_path = os.path.join(param_save_folder, "epoch_{}_seed_{}.pth".format(epoch, self.args.seed))
                torch.save(self.model.state_dict(), param_save_path)
        
        
    def evaluate(self):
        total_correct = 0
        total_loss = 0.0
        self.model.eval()
        for images, labels in self.test_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = self.model(images)
            pred_labels = torch.argmax(outputs, dim=1)
            loss = self.criterion(outputs, labels)
            
            matches = pred_labels.eq(labels).float()
            correct = matches.sum().item()
            
            total_correct += correct
            total_loss += loss.item() * images.size(0)
            
        accuracy = total_correct / len(self.test_loader.dataset)
        loss = total_loss / len(self.test_loader.dataset)
        return accuracy, loss
        