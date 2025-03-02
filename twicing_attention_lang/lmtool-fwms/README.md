## Code for Language Modeling Task in Our Paper

## Requirements
This toolkit requires PyTorch `torch` and Ninja `ninja` (to compile the cuda kernels).

The experiments for the paper were conducted with Python 3.6 and PyTorch >= 1.4.0.

The toolkit supports [Weights & Biases](https://docs.wandb.ai/) for monitoring jobs. If you use it, also install `wandb`.

## Instructions

Run `sh getdata.sh` to download the data.

### Training

Run following commands to reproduce our results for WikiText-103 language modeling.
```sh
bash run_sharp_small.sh
```