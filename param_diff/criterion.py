import torch
import torch.nn.functional as F


def kld_loss(latent):
        mu = torch.mean(latent, dim=1)
        log_var = torch.log(torch.var(latent, dim=1) + 1e-8)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        return kld_loss