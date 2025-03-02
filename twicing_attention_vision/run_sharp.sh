# Train Script for DeiT-Twicing
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 29540 \
--use_env main.py --model deit_sharp_tiny_patch16_224 --batch-size 256 --data-path /sharpformer/data/imagenet  \
--output_dir /sharpformer/checkpoints/twicing-04 --project_name 'attentionpp-imagenet' --job_name twicing-04 --seed 0 --use_wandb