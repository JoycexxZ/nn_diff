import logging
import math
import os
import time

import diffusers
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from tqdm import tqdm
from torchvision import datasets, transforms
import argparse

from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler

from modules.unet import AE_CNN_bottleneck
from param_dataset import ParamDataset
from modules.unet_1d import naive_unet_1d
from diff_pipeline import WeightDiffPipeline

import sys
sys.path.append("../param_gen/models")
from convnet import ConvNet2_mnist, ConvNet3_cifar



logger = get_logger(__name__, log_level="INFO")


def test(model, test_loader):
    total_correct = 0
    model.eval()
    for images, labels in test_loader:
        images, labels = images, labels
        outputs = model(images)
        pred_labels = torch.argmax(outputs, dim=1)

        matches = pred_labels.eq(labels).float()
        correct = matches.sum().item()

        total_correct += correct

    accuracy = total_correct / len(test_loader.dataset)
    return accuracy
 
 
def main(args):

    output_dir = "../outputs/diff_training/{}/{}".format(args.name, args.exp_name)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # output_dir=f"{output_dir}/acc_log",
        log_with="wandb",
    )
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(filename)s: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        # transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        # transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
        
    # data
    train_batch_size = args.train_batch_size
    train_dataset = ParamDataset(args.data, augmentation=True)
    data_dim = train_dataset.get_data_dim()
    logger.info("param dim: {}".format(data_dim))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Handle the output folder creation
    if accelerator.is_main_process:
        # output_log = "../outputs/diff_training/{}/{}/log.txt".format(args.name, args.exp_name)
        # if os.path.exists(output_log):
        #     output_log = "../outputs/task_training/{}/{}/log_seed{}_{}.txt".format(args.name, args.exp_name, args.seed, time.strftime("%Y%m%d-%H%M%S"))
        if os.path.exists(output_dir):
            output_dir = "../outputs/diff_training/{}/{}_{}".format(
                args.name, args.exp_name, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(output_dir, exist_ok=True)

    # Models
    if args.unet.model == "AE_CNN_bottleneck":
        unet = AE_CNN_bottleneck(
            in_dim=data_dim, 
            in_channel=args.unet.in_channel,
            time_step=args.scheduler.n_timestep,
        )
    elif args.unet.model == "naive_unet1d":
        unet = naive_unet_1d(
            input_dim=data_dim, 
        )
    else:
        raise ValueError("Model not implemented")

    noise_scheduler = DDPMScheduler(
        beta_start=args.scheduler.start,
        beta_end=args.scheduler.end,
        num_train_timesteps=args.scheduler.n_timestep,
        beta_schedule=args.scheduler.schedule,
    )
    val_scheduler = DDPMScheduler(
        beta_start=args.scheduler.start,
        beta_end=args.scheduler.end,
        num_train_timesteps=args.scheduler.n_timestep,
        beta_schedule=args.scheduler.schedule,
    )

    # optimizer
    learning_rate = args.learning_rate
    if args.scale_lr:
        learning_rate = learning_rate * args.gradient_accumulation_steps * train_batch_size * accelerator.num_processes

    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )

    # lr scheduler
    num_training_steps = args.num_train_steps
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.num_train_steps * args.gradient_accumulation_steps,
    )
    
    # test loader
    if args.test.dataset == 'mnist':
        normal_value = {'mean': [0.1307,], 'std': [0.3081,]}
    elif args.test.dataset == 'cifar':
        normal_value = {"mean": [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    test_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(normal_value["mean"], normal_value["std"])
                                ])
    if args.test.dataset == "mnist":
        test_dataset = datasets.MNIST(root=args.test.root, train=False, download=True, transform=test_transform)
    elif args.test.dataset == "cifar":
        test_dataset = datasets.CIFAR10(root=args.test.root, train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test.test_bs, shuffle=False)
    
    # test model
    if args.data.dataset == "mnist_resnet18":
        test_model = train_dataset.get_model().cuda()
        train_layers = train_dataset.get_layer_names()
    elif args.test.model == "ConvNet3":
        test_model = ConvNet3_cifar(3).cuda()
    elif args.test.model == "ConvNet2":
        test_model = ConvNet2_mnist(1).cuda()


    # validation pipeline here to finish
    pipeline = WeightDiffPipeline(
        vae=None,
        unet=unet,
        scheduler=val_scheduler,
    )

    # prepare with accelerator
    unet, optimizer, train_dataloader, lr_scheduler, test_loader, test_model = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler,test_loader, test_model
    )

    # mixed precision training should cast the un-trained model to half precision here

    # train
    num_training_epochs = math.ceil(num_training_steps / len(train_dataloader))

    if accelerator.is_main_process:
        accelerator.init_trackers("nn_diffusion-training", init_kwargs={"wandb": {"name": f"{args.name}_{args.exp_name}"}})

    global_step = 0
    progress_bar = tqdm(range(global_step, num_training_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("steps")
    
    test_model.eval()

    for epoch in range(num_training_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                weight_values = batch["weight_value"] # [bs, dim]
                bs = weight_values.size(0)
                # noise
                noise = torch.randn_like(weight_values) # [bs, dim]
                # timestep
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,)) # [bs]
                timesteps = timesteps.long()
                
                noisy_weight_values = noise_scheduler.add_noise(weight_values, noise, timesteps)
                
                if noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(weight_values, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")
                
                # print(noisy_weight_values.device, timesteps.device, target.device)
                # print(noisy_weight_values.shape, timesteps.shape, target.shape)
                model_pred = unet(noisy_weight_values, timesteps.to(noisy_weight_values.device))
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                avg_loss = accelerator.gather(loss.repeat(bs)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # Save here
                # if global_step % args.checkpoint_steps == 0:

                # test here
                if global_step % args.validation_steps == 0:
                    weight_sample = pipeline(num_inference_steps=50, latents=None, batch_size=5)
                    _test_acc = 0.0
                    for i in range(5):
                        sample = weight_sample.model_weights[i]
                        test_model.load_param_from_tensor(sample)
                        _test_acc += test(test_model, test_loader)
                    _test_acc = _test_acc / 5
                    accelerator.log({"test_accuracy": _test_acc}, step=global_step)
                


            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= num_training_steps:
                break

    accelerator.wait_for_everyone()
    # create the pipeline and save

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../configs/diff/param_diff.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)