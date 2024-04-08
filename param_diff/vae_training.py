import torch
import os
import accelerate
import logging
import diffusers
import math
import argparse

from accelerate.logging import get_logger
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from accelerate.utils import set_seed
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import datasets, transforms
from omegaconf import OmegaConf

from param_dataset import ParamDataset, ParamDataset2
from modules.vae_encoder import small, medium
from modules.vae_encoder_cnn import Latent_AE_cnn_big
from criterion import kld_loss

import sys
sys.path.append("../param_gen/models")
from convnet import ConvNet2_mnist, ConvNet3_cifar
from resnet import resnet_load_param_from_tensor

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
    output_dir = "../outputs/vae_training/{}/{}".format(args.name, args.exp_name)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # output_dir="{}/acc_log".format(output_dir),
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

    if args.ae.model == "medium":
        autoencoder = medium(in_dim=data_dim,
                             input_noise_factor=args.data.augmentation_scale,
                             latent_noise_factor=args.ae.latent_augmentation)
    elif args.ae.model == "latent_ae_cnn_big":
        autoencoder = Latent_AE_cnn_big(in_dim=data_dim)
        
    if args.data.dataset == "mnist_resnet18":
        test_model = train_dataset.get_model().cuda()
        train_layers = train_dataset.get_layer_names()
    elif args.test.model == "ConvNet3":
        test_model = ConvNet3_cifar(3).cuda()
    elif args.test.model == "ConvNet2":
        test_model = ConvNet2_mnist(1).cuda()

    # optimizer
    learning_rate = args.learning_rate
    if args.scale_lr:
        learning_rate = learning_rate * args.gradient_accumulation_steps * train_batch_size * accelerator.num_processes

    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        autoencoder.parameters(),
        lr=learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_eps
    )

    # lr scheduler
    num_training_steps = args.num_training_steps
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=num_training_steps * args.gradient_accumulation_steps,
    )
    
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

    # prepare with accelerator
    autoencoder, optimizer, train_dataloader, lr_scheduler, test_model, test_loader = accelerator.prepare(
        autoencoder, optimizer, train_dataloader, lr_scheduler, test_model, test_loader
    )

    # train
    num_training_epochs = math.ceil(num_training_steps / len(train_dataloader))

    if accelerator.is_main_process:
        accelerator.init_trackers("vae-training", init_kwargs={"wandb": {"name": f"{args.name}_{args.exp_name}"}})

    global_step = 0
    progress_bar = tqdm(range(global_step, num_training_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("steps")

    test_model.eval()

    for epoch in range(num_training_epochs):
        autoencoder.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(autoencoder):
                weight_values = batch["weight_value"]  # [bs, dim]
                bs = weight_values.size(0)
                # model_pred = autoencoder(weight_values)
                _weight_values = (1-args.data.augmentation_scale) * weight_values + args.data.augmentation_scale * torch.randn_like(weight_values)
                latent = autoencoder.Enc(_weight_values)
                noisy_latent = (1-args.ae.latent_augmentation)*latent + args.ae.latent_augmentation * torch.randn_like(latent)
                noisy_latent = torch.clamp(noisy_latent, -1, 1)
                model_pred = autoencoder.Dec(noisy_latent)
                # print(model_pred.shape, weight_values.shape)

                loss = F.mse_loss(model_pred.float(), weight_values.float(), reduction="sum")
                loss += args.kl_loss_weight * kld_loss(latent)

                avg_loss = accelerator.gather(loss.repeat(bs)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(autoencoder.parameters(), args.max_grad_norm)
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
                    if accelerator.is_main_process:
                        _test_acc = 0.0
                        _best_acc = 0.0
                        test_batch = weight_values[:5]
                        test_output = autoencoder.Dec(autoencoder.Enc(test_batch))
                        for sample in range(test_output.size(0)):
                            test_model.load_param_from_tensor(test_output[sample])
                            # resnet_load_param_from_tensor(model_pred[sample], train_layers, test_model)
                            sample_test_acc = test(test_model, test_loader)
                            _test_acc += sample_test_acc
                            if sample_test_acc > _best_acc:
                                _best_acc = sample_test_acc
                        _test_acc = _test_acc / 5
                        accelerator.log({"test_accuracy": _test_acc}, step=global_step)
                        accelerator.log({"best_accuracy": _best_acc}, step=global_step)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= num_training_steps:
                break

    accelerator.wait_for_everyone()
    # create the pipeline and save

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../configs/diff/vae_training.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)