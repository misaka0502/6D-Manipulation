#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mr_rppo_low

python -m src.train.residual_ppo +experiment=rl/residual_ppo \
    base_policy.wandb_id=mr-state-dr-low-1/qh11vha4 \
    base_policy.wt_type=best_test_loss \
    env.task=mug_rack \
    env.randomness=low \
    num_env_steps=400 \
    num_envs=1024 \
    n_iterations_train_only_value=20 \
    update_epochs=25 \
    eval_interval=20 \
    checkpoint_interval=100 \
    wandb.project=mr-rppo-dr-low-1 \
    debug=false 
