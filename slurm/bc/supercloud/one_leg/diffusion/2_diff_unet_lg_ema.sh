#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=2_diff_unet_lg
#SBATCH -c 20

python -m src.train.bc +experiment=state/diff_unet \
    wandb.mode=offline \
    wandb.entity=robust-assembly \
    training.ema.use=true \
    dryrun=false