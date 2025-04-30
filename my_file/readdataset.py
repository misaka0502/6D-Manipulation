from src.dataset.dataset import (
    FurnitureImageDataset,
)
from src.common.files import get_processed_paths
from omegaconf import DictConfig, OmegaConf
import hydra
from src.dataset import get_normalizer
from src.dataset.normalizer import Normalizer
from PIL import Image
from torchvision.transforms import ToPILImage
import numpy as np
OmegaConf.register_new_resolver("eval", eval)
import time
def to_native(obj):
    try:
        return OmegaConf.to_object(obj)
    except ValueError:
        return obj

def set_dryrun_params(config: DictConfig):
    if config.dryrun:
        OmegaConf.set_struct(config, False)
        config.training.steps_per_epoch = 1
        config.data.data_subset = 1

        if config.rollout.rollouts:
            config.rollout.every = 1
            config.rollout.num_rollouts = 1
            config.rollout.loss_threshold = float("inf")
            config.rollout.max_steps = 10

        config.wandb.mode = "disabled"

        OmegaConf.set_struct(config, True)

# @hydra.main(config_path="./src/config", config_name="base")
def main():
    # set_dryrun_params(config)
    # OmegaConf.resolve(config)
    data_path = get_processed_paths(
        environment=to_native("sim"),
        task=to_native("one_leg"),
        demo_source=to_native("teleop"),
        randomness=to_native("low"),
        demo_outcome=to_native("success"),
    )
    normalizer: Normalizer = get_normalizer(
        "min_max", "delta"
    )
    dataset = FurnitureImageDataset(
        dataset_paths=data_path,
        pred_horizon=32,
        obs_horizon=1,
        action_horizon=8,
        normalizer=normalizer.get_copy(),
        augment_image=True,
        data_subset=None,
        control_mode="pos",
        first_action_idx=0,
        pad_after=True,
        max_episode_count=None,
    )
    save_path = "./poses"
    # print(dataset[0]['parts_poses'].shape)
    # time.sleep(10000)
    poses_1 = []
    poses_2 = []
    poses_3 = []
    poses_4 = []
    poses_5 = []
    poses_6 = []
    # print(len(dataset))
    # time.sleep(10000)
    for i in range(500):
        print(i)
        poses_1.append(dataset[i]['parts_poses'].squeeze().numpy()[:7])
        poses_2.append(dataset[i]['parts_poses'].squeeze().numpy()[7:14])
        poses_3.append(dataset[i]['parts_poses'].squeeze().numpy()[14:21])
        poses_4.append(dataset[i]['parts_poses'].squeeze().numpy()[21:28])
        poses_5.append(dataset[i]['parts_poses'].squeeze().numpy()[28:35])
        poses_6.append(dataset[i]['parts_poses'].squeeze().numpy()[35:42])

    np.savetxt(f"{save_path}/poses_{1}.txt", poses_1)
    np.savetxt(f"{save_path}/poses_{2}.txt", poses_2)
    np.savetxt(f"{save_path}/poses_{3}.txt", poses_3)
    np.savetxt(f"{save_path}/poses_{4}.txt", poses_4)
    np.savetxt(f"{save_path}/poses_{5}.txt", poses_5)
    np.savetxt(f"{save_path}/poses_{6}.txt", poses_6)
    # print(dataset[0].keys())
    # to_pil = ToPILImage()
    # img = to_pil(dataset[0]['color_image1'].squeeze(0).permute(2, 0, 1))
    # img.save('color_image1.png')
    # img = to_pil(dataset[0]['color_image2'].squeeze(0).permute(2, 0, 1))
    # img.save('color_image2.png')
    # if dataset[0]['depth_image1'] is not None:
    #     img = to_pil(dataset[0]['depth_image1'].squeeze(0).permute(2, 0, 1))
    #     img.save('depth_image1.png')
    #     img = to_pil(dataset[0]['depth_image2'].squeeze(0).permute(2, 0, 1))
    #     img.save('depth_image2.png')
    # print(len(dataset))
    # print(dataset[0]['parts_poses'].shape)
    # print(dataset[0]['parts_poses'])

if __name__ == "__main__":  
    main()