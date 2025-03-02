# ImagetNet with DeiT transformer

## Installation
- Python>=3.7
- Requirements:
```bash
pip install -r requirements.txt
```

## Compile CUDA code

In `./fourier_layer-extension`, run `python setup_cuda.py install` to compile the CUDA code before running the training.

## Experiments

Run `run_sharp.sh` script for training.