#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-h100,vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=01-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=4_only_camera_1_prop_drop

python -m src.train.bc +experiment=image/real_ol_cotrain \
    actor/diffusion_model=transformer \
    training.actor_lr=1e-4 \
    training.encoder_lr=1e-6 \
    training.num_epochs=5000 \
    demo_source=teleop \
    task=pick_cup \
    regularization.front_camera_dropout=1.0 \
    regularization.proprioception_dropout=0.2 \
    randomness=low \
    environment=real \
    wandb.entity=dexterity-hub \
    wandb.project=pick-cup-1 \
    wandb.name=one-camera-prop-drop-1 \
    dryrun=false