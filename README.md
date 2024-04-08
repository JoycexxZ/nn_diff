# NN Diffusion Project

## Code running

- **Step 1:** generate parameters (training set raw data)

    Configure file `configs/task/mnist_gen.yaml` for seed settings

    Configure file `configs/task/mnist.yaml` for training hyperparameters

    ```bash
    cd param_gen
    wandb sweep ../configs/task/mnist_gen.yaml
    ```

- **Step 2**: diffusion training
    Configure file `configs/diff/vae_training.yaml` for VAE training.
    Configure file `configs/diff/param_diff.yaml` for diffusion training.
