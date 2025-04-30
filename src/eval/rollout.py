from gymnasium import Env
from omegaconf import DictConfig  # noqa: F401
import torch

import collections

import numpy as np
from tqdm import tqdm, trange
from ipdb import set_trace as bp  # noqa: F401

from typing import Dict, Optional, Union
from pathlib import Path

from src.behavior.base import Actor
from src.visualization.render_mp4 import create_in_memory_mp4
from src.common.context import suppress_all_output
from src.common.tasks import task2idx
from src.common.files import get_processed_path, trajectory_save_dir
from src.data_collection.io import save_raw_rollout
from src.data_processing.utils import filter_and_concat_robot_state
from src.data_processing.utils import resize, resize_crop
from tensordict import TensorDict

from copy import deepcopy

import wandb
import zarr

from FoundationPose.estimater import *
from FoundationPose.datareader import *
from FoundationPose.Utils import *
from FoundationPose.learning.training.predict_score import *
from FoundationPose.learning.training.predict_pose_refine import *

from furniture_bench.config import config
from furniture_bench.utils.pose import get_mat
import furniture_bench.controllers.control_utils as C
from scipy.spatial.transform import Rotation as R
import cv2
from datetime import datetime

RolloutStats = collections.namedtuple(
    "RolloutStats",
    [
        "success_rate",
        "n_success",
        "n_rollouts",
        "epoch_idx",
        "rollout_max_steps",
        "total_return",
        "total_reward",
    ],
)

RolloutSaveValues = collections.namedtuple(
    "RolloutSaveValues",
    [
        "robot_states",
        "imgs1",
        "imgs2",
        "actions",
        "rewards",
        "parts_poses",
        "parts_poses_est"
    ],
)

ROBOT_HEIGHT = 0.015
table_pos = np.array([0.8, 0.8, 0.4])
table_half_width = 0.015
table_surface_z = table_pos[2] + table_half_width
franka_pose = np.array(
    [0.5 * -table_pos[0] + 0.1, 0, table_surface_z + ROBOT_HEIGHT]
)
base_tag_from_robot_mat = get_mat(
    (0.23 + 0.0715, 0, -ROBOT_HEIGHT), (np.pi, 0, np.pi / 2)
)
franka_from_origin_mat = get_mat(
    [franka_pose[0], franka_pose[1], franka_pose[2]],
    [0, 0, 0],
)
save_flag = True
def sim_to_april_mat():
    return torch.tensor(
        np.linalg.inv(base_tag_from_robot_mat) @ np.linalg.inv(franka_from_origin_mat),
        device="cpu", dtype=torch.float64
    )

def sim_coord_to_april_coord(sim_coord_mat):
    return sim_to_april_mat() @ sim_coord_mat

class coordinate_transform:
    def __init__(self, cam_pos, cam_target, init_pose=None):
        self.revise = False
        self.init_pose = init_pose
        self.r = None
        cam_pos = np.array(cam_pos)
        cam_target = np.array(cam_target)
        z_camera = (cam_target - cam_pos) / np.linalg.norm(cam_target - cam_pos)
        up_axis = np.array([0, 0, 1])  # Assuming Z is the up axis
        x_camera = -np.cross(up_axis, z_camera)
        x_camera /= np.linalg.norm(x_camera)
        y_camera = np.cross(z_camera, x_camera)
        R_camera_sim = np.vstack([x_camera, y_camera, z_camera]).T
        self.T_camera_sim = np.eye(4)
        self.T_camera_sim[:3, :3] = R_camera_sim
        self.T_camera_sim[:3, 3] = cam_pos
    def cam_coord_to_april_coord(self, pose_est_cam, pose_gt = None):
        pos_est_sim = self.T_camera_sim @ pose_est_cam
        pose_est_april_coord = np.concatenate(
            [
                *C.mat2pose(
                    sim_coord_to_april_coord(
                        torch.tensor(pos_est_sim, device="cpu", dtype=torch.float64)
                    )
                )
            ]
        )
        if self.revise is False and self.init_pose is not None:
            R1 = R.from_quat(self.init_pose[-4:]).as_matrix()
            R2 = R.from_quat(pose_est_april_coord[-4:]).as_matrix()
            R_relative = np.dot(R2, R1.T)
            trace = np.trace(R_relative)
            angle = np.arccos((trace - 1) / 2)
            angle = np.degrees(angle)
            print(f"angle error: {angle}")
            if self.r is None:
                if 80. < angle < 100.:
                    print(f"angle error111:{angle}")
                    temp_r = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
                    self.r = temp_r
                    R3 = R2 @ self.r
                    R_relative = np.dot(R3, R1.T)
                    trace = np.trace(R_relative)
                    angle = np.arccos((trace - 1) / 2)
                    angle = np.degrees(angle)
                    while(angle > 10.):
                        print(f"angle error round1: {angle}")
                        self.r = self.r @ temp_r
                        R3 = R2 @ self.r
                        R_relative = np.dot(R3, R1.T)
                        trace = np.trace(R_relative)
                        angle = np.arccos((trace - 1) / 2)
                        angle = np.degrees(angle)
                        print(f"angle error round2: {angle}")
                elif 170. < angle < 190.:
                    print(f"angle error222:{angle}")
                    self.r = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
                    if pose_gt is not None:
                        print(pose_est_april_coord)
                        print(pose_gt)
            self.revise = True
        if self.r is not None:
            R1 = R.from_quat(self.init_pose[-4:]).as_matrix()
            R2 = R.from_quat(pose_est_april_coord[-4:]).as_matrix()
            R2 = R2 @ self.r
            R_relative = np.dot(R2, R1.T)
            trace = np.trace(R_relative)
            angle = np.arccos((trace - 1) / 2)
            angle = np.degrees(angle)
            pose_est_april_coord[-4:] = C.mat2quat(torch.tensor(R2, device="cpu", dtype=torch.float64))
            print(f"angle error revised:{angle}")
        # pose_est_april_coord[np.abs(pose_est_april_coord) < 0.0015] = 0
        return pose_est_april_coord

def resize_image(obs, key):
    try:
        obs[key] = resize(obs[key])
    except KeyError:
        pass


def resize_crop_image(obs, key):
    try:
        obs[key] = resize_crop(obs[key])
    except KeyError:
        pass


def squeeze_and_numpy(d: Dict[str, Union[torch.Tensor, np.ndarray, float, int, None]]):
    """
    Recursively squeeze and convert tensors to numpy arrays
    Convert scalars to floats
    Leave NoneTypes alone
    """
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = squeeze_and_numpy(v)

        elif v is None:
            continue

        elif isinstance(v, (torch.Tensor, np.ndarray)):
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            d[k] = v.squeeze()

        else:
            raise ValueError(f"Unsupported type: {type(v)}")

    return d


def tensordict_to_list_of_dicts(tensordict):
    list_of_dicts = []
    keys = list(tensordict.keys())
    num_elements = tensordict[keys[0]].shape[0]

    for i in range(num_elements):
        dict_element = {}
        for key in keys:
            dict_element[key] = tensordict[key][i].cpu().numpy()
        list_of_dicts.append(dict_element)

    return list_of_dicts


class SuccessTqdm(tqdm):
    def __init__(
        self,
        num_envs: int,
        n_rollouts: int,
        task_name: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.num_envs = num_envs
        self.n_rollouts = n_rollouts
        self.task_name = task_name
        self.round = 0
        self.success_in_prev_rounds = 0

    def pbar_desc(self, n_success: int):
        total = self.round * self.num_envs
        n_success += self.success_in_prev_rounds
        success_rate = n_success / total if total > 0 else 0
        self.set_description(
            f"Performing rollouts ({self.task_name}): "
            f"round {self.round}/{self.n_rollouts//self.num_envs}, "
            f"success: {n_success}/{total} ({success_rate:.1%})"
        )

    def before_round(self, n_success: int):
        self.success_in_prev_rounds = n_success
        self.round += 1

        self.pbar_desc(0)

save_vis_flag = True
def rollout(
    env: Env,
    actor: Actor,
    rollout_max_steps: int,
    pbar: SuccessTqdm = None,
    resize_video: bool = True,
    n_parts_assemble: int = 1,
    save_rollouts: bool = False,
    ests:FoundationPose = None,
    readers:YcbineoatReader = None,
    to_origins = None,
    bboxs = None,
    debug = 0,
    debug_dir = None,
    use_pose_est: bool = False,
    num_poses=1,
    iter: int=0
) -> Optional[RolloutSaveValues]:
    # get first observation
    with suppress_all_output(False):
        obs = env.reset()
        actor.reset()

    video_obs = deepcopy(obs)

    # Resize the images in the observation if they exist
    # resize_image(obs, "color_image1")
    # resize_crop_image(obs, "color_image2")

    if resize_video:
        resize_image(video_obs, "color_image1")
        resize_crop_image(video_obs, "color_image2")
    init_pose_ref=obs["parts_poses"]
    # save visualization and rewards
    robot_states = [TensorDict(video_obs["robot_state"], batch_size=env.num_envs)]
    imgs1 = [] if "color_image1" not in video_obs else [video_obs["color_image1"].cpu()]
    imgs2 = [] if "color_image2" not in video_obs else [video_obs["color_image2"].cpu()]
    parts_poses = [video_obs["parts_poses"].cpu()]
    parts_poses_est ={}
    actions = list()
    rewards = torch.zeros((env.num_envs, rollout_max_steps), dtype=torch.float32)
    done = torch.zeros((env.num_envs, 1), dtype=torch.bool, device="cuda")

    step_idx = 0

    if readers is not None:
        global save_vis_flag
        os.makedirs(f'{debug_dir}/rollouts_vis/leg', exist_ok=True)
        os.makedirs(f'{debug_dir}/rollouts_vis/top', exist_ok=True)
        camera_idx = ["5", "4"]
        with suppress_all_output(True):
            colors = [readers[i].get_color(obs[f"color_image{camera_idx[i]}"].squeeze(0).cpu().numpy()) for i in range(num_poses)]
            depths = [readers[i].get_depth(65535-(obs[f"depth_image{camera_idx[i]}"].squeeze(0).cpu().numpy() * 1000).astype(np.uint16)) for i in range(num_poses)]
            masks = [readers[i].get_mask(9).astype(bool) for i in range(num_poses)]
            poses_begin = [ests[i].register(K=readers[i].K, rgb=colors[i], depth=depths[i], ob_mask=masks[i], iteration=5) for i in range(num_poses)]

            if debug >= 1:
                center_poses = [poses_begin[i]@np.linalg.inv(to_origins[i]) for i in range(num_poses)]
                vis_begins = [draw_posed_3d_box(readers[i].K, img=colors[i], ob_in_cam=center_poses[i], bbox=bboxs[i]) for i in range(num_poses)]
                # center_pose = pose_begin@np.linalg.inv(to_origin)
                # center_pose2 = pose_begin2@np.linalg.inv(to_origin2)
                # vis_begin = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
                # vis_begin = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
                cv2.imwrite(f'{debug_dir}/rollouts_vis/leg/{iter:019d}_begin.png', vis_begins[0])
                # vis_begin2 = draw_posed_3d_box(reader2.K, img=color2, ob_in_cam=center_pose2, bbox=bbox2)
                # vis_begin2 = draw_xyz_axis(color2, ob_in_cam=center_pose2, scale=0.1, K=reader2.K, thickness=3, transparency=0, is_input_rgb=True)
                cv2.imwrite(f'{debug_dir}/rollouts_vis/top/{iter:019d}_begin.png', vis_begins[1])
    if readers is not None:
        leg_init_pose_ref=obs["parts_poses"][:, -14:-7].squeeze(0).cpu().numpy()
        top_init_pose_ref=obs["parts_poses"][:, :7].squeeze(0).cpu().numpy()
        leg_coord_trans = coordinate_transform([-0.5, 0.6, 0.8], [0.3, -0.00, 0.3], init_pose=leg_init_pose_ref)
        top_coord_trans = coordinate_transform([0.3, -0.65, 0.8], [0.3, 0.8, 0.00], init_pose=top_init_pose_ref)
        leg_pose_begin_april_coord = leg_coord_trans.cam_coord_to_april_coord(poses_begin[0])
        print(f"leg pose: {obs['parts_poses'][:, -14:-7]}")
        print(f"leg pose est: {leg_pose_begin_april_coord}")
        obs["parts_poses"][:, -14:-7] = torch.tensor(leg_pose_begin_april_coord, device=env.device)
        top_pose_begin_april_coord = top_coord_trans.cam_coord_to_april_coord(poses_begin[1], obs["parts_poses"][:, :7])
        print(f"top pose: {obs['parts_poses'][:, :7]}")
        print(f"top pose est: {top_pose_begin_april_coord}")
        obs["parts_poses"][:, :7] = torch.tensor(top_pose_begin_april_coord, device=env.device)
        parts_poses_est["leg"] = [leg_pose_begin_april_coord]
        parts_poses_est["top"] = [top_pose_begin_april_coord]
        # print("leg SE3:")
        # print(poses_begin[0])
        # print("top SE3:")
        # print(poses_begin[1])
    # time.sleep(10000)

    # TODO - figure out how to fix this
    actor.normalizer = actor.normalizer.to(actor.device)
    actor.model = actor.model.to(actor.device)

    while not done.all():
        # Convert from robot state dict to robot state tensor
        obs["robot_state"] = env.filter_and_concat_robot_state(obs["robot_state"])

        # Get the next actions from the actor
        # print(f"leg pose: {obs['parts_poses'][:, -14:-7]}")
        # print(f"top pose: {obs['parts_poses'][:, :7]}")
        action_pred = actor.action(obs)
        # action_pred = torch.tensor(actions[step_idx], device="cuda").unsqueeze(0)
        # action_pred = actor.normalizer(action_pred, "action", forward=False)

        obs, reward, done, _ = env.step(action_pred, sample_perturbations=False)
        # print(f"leg pose gt: {obs['parts_poses'][:, -14:-7]}")
        # print(f"top pose gt: {obs['parts_poses'][:, :7]}")
        video_obs = deepcopy(obs)

        if readers is not None:
            obs["parts_poses"] = init_pose_ref
            with suppress_all_output(True):
                colors = [readers[i].get_color(obs[f"color_image{camera_idx[i]}"].squeeze(0).cpu().numpy()) for i in range(num_poses)]
                depths = [readers[i].get_depth(65535-(obs[f"depth_image{camera_idx[i]}"].squeeze(0).cpu().numpy() * 1000).astype(np.uint16)) for i in range(num_poses)]
                poses = [ests[i].track_one(rgb=colors[i], depth=depths[i], K=readers[i].K, iteration=2) for i in range(num_poses)]

                if save_vis_flag:
                    center_poses = [poses[i]@np.linalg.inv(to_origins[i]) for i in range(num_poses)]
                    vis = [draw_posed_3d_box(readers[i].K, img=colors[i], ob_in_cam=center_poses[i], bbox=bboxs[i]) for i in range(num_poses)]
                    imageio.imwrite(f'{debug_dir}/track_vis/leg/{step_idx:019d}.png', vis[0])
                    imageio.imwrite(f'{debug_dir}/track_vis/top/{step_idx:019d}.png', vis[1])
                if debug>=2:
                    center_poses = [poses[i]@np.linalg.inv(to_origins[i]) for i in range(num_poses)]
                    vis = [draw_posed_3d_box(readers[i].K, img=colors[i], ob_in_cam=center_poses[i], bbox=bboxs[i]) for i in range(num_poses)]
                    # cv2.imshow('1', vis[...,::-1])
                    # cv2.waitKey(1)
                if debug>=3:
                    os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
                    imageio.imwrite(f'{debug_dir}/track_vis/leg/{step_idx:019d}.png', vis[0])
                    os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
                    imageio.imwrite(f'{debug_dir}/track_vis/top/{step_idx:019d}.png', vis[1])
        
                leg_pose_april_coord = leg_coord_trans.cam_coord_to_april_coord(poses[0])
                # print(f"pose: {obs['parts_poses'][:, -14:-7]}")
                obs["parts_poses"][:, -14:-7] = torch.tensor(leg_pose_april_coord, device=env.device)
                top_pose_april_coord = top_coord_trans.cam_coord_to_april_coord(poses[1])
                # print(obs["parts_poses"][:, :7])
                obs["parts_poses"][:, :7] = torch.tensor(top_pose_april_coord, device=env.device)
            # print(f"leg pose est: {leg_pose_april_coord}")
            # print(f"top pose est: {top_pose_april_coord}")
                # print(poses[1])
        
        # if save_flag:
        #     cv2.imwrite(f'{debug_dir}/rollouts_vis/leg/rgb/{step_idx:019d}.png', obs["color_image2"].squeeze(0).cpu().numpy())

        # Resize the images in the observation if they exist
        # resize_image(obs, "color_image1")
        # resize_crop_image(obs, "color_image2")

        # Save observations for the policy
        if resize_video:
            resize_image(video_obs, "color_image1")
            resize_crop_image(video_obs, "color_image2")
        parts_poses.append(video_obs["parts_poses"].cpu())
        if readers is not None:
            parts_poses_est["leg"].append(leg_pose_april_coord)
            parts_poses_est["top"].append(top_pose_april_coord)        
        # Store the results for visualization and logging
        if save_rollouts:
            robot_states.append(
                TensorDict(video_obs["robot_state"], batch_size=env.num_envs)
            )
            # if "color_image1" in video_obs:
            #     imgs1.append(video_obs["color_image1"].cpu())
            # if "color_image2" in video_obs:
            #     imgs2.append(video_obs["color_image2"].cpu())
            actions.append(action_pred.cpu())
            parts_poses.append(video_obs["parts_poses"].cpu())

        # Always store rewards as they are used to calculate success
        rewards[:, step_idx] = reward.squeeze().cpu()

        # update progress bar
        step_idx += 1
        if pbar is not None:
            pbar.set_postfix(step=step_idx)
            n_success = (rewards.sum(dim=1) == n_parts_assemble).sum().item()
            pbar.pbar_desc(n_success)
            pbar.update()

        if step_idx >= rollout_max_steps:
            done = torch.ones((env.num_envs, 1), dtype=torch.bool, device="cuda")

        if done.all():
            save_vis_flag = False
            if readers is not None:
                center_poses = [poses[i]@np.linalg.inv(to_origins[i]) for i in range(num_poses)]
                vis_ends = [draw_posed_3d_box(readers[i].K, img=colors[i], ob_in_cam=center_poses[i], bbox=bboxs[i]) for i in range(num_poses)]
                cv2.imwrite(f'{debug_dir}/rollouts_vis/leg/{iter:019d}_end.png', vis_ends[0])
                cv2.imwrite(f'{debug_dir}/rollouts_vis/top/{iter:019d}_end.png', vis_ends[1])
            break

    return RolloutSaveValues(
        torch.stack(robot_states, dim=1) if robot_states else [],
        torch.stack(imgs1, dim=1) if imgs1 else [],
        torch.stack(imgs2, dim=1) if imgs2 else [],
        torch.stack(actions, dim=1) if actions else [],
        rewards,
        torch.stack(parts_poses, dim=1) if parts_poses else [],
        parts_poses_est if parts_poses else []
    )


@torch.no_grad()
def calculate_success_rate(
    env: Env,
    actor: Actor,
    n_rollouts: int,
    rollout_max_steps: int,
    epoch_idx: int,
    discount: float = 0.99,
    rollout_save_dir: Optional[Path] = None,
    save_rollouts_to_wandb: bool = False,
    save_failures: bool = False,
    n_parts_assemble: Optional[int] = None,
    compress_pickles: bool = False,
    resize_video: bool = True,
    n_steps_padding: int = 30,
    break_on_n_success: bool = False,
    stop_after_n_success: int = 0,
    record_first_state_only: bool = False,
    use_pose_est: bool = False,
    num_poses = 1
) -> RolloutStats:

    pbar = SuccessTqdm(
        num_envs=env.num_envs,
        n_rollouts=n_rollouts,
        task_name=env.task_name,
        total=rollout_max_steps * (n_rollouts // env.num_envs),
        desc="Performing rollouts",
        leave=True,
        unit="step",
    )

    if n_parts_assemble is None:
        n_parts_assemble = env.n_parts_assemble

    tbl = wandb.Table(
        columns=["rollout", "success", "epoch", "reward", "return", "steps"]
    )

    n_success = 0

    all_robot_states = list()
    all_imgs1 = list()
    all_imgs2 = list()
    all_actions = list()
    all_rewards = list()
    all_parts_poses = list()
    all_success = list()

    save_rollouts = rollout_save_dir is not None or save_rollouts_to_wandb

    if use_pose_est:
        scorer:ScorePredictor = None
        refiner:PoseRefinePredictor = None
        glctx:dr.RasterizeCudaContext = None
        est:FoundationPose = None
        reader:YcbineoatReader = None
        debug_dir:str = None
        mesh = None
        to_origin = None
        bbox = None
        debug = 1
        time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        code_dir = "/home2/zxp/Projects/robust-rearrangement"
        debug_dir = f'{code_dir}/FoundationPose/debug/{time_now}'
        os.system(f'rm -rf {debug_dir}/rollouts_vis/* {debug_dir}/rollouts_ob/* && mkdir -p {debug_dir}/rollouts_vis {debug_dir}/rollouts_ob')
        pbar.pbar_desc(n_success)
        set_logging_format()
        set_seed(0)
        os.system(f'mkdir -p {debug_dir}/track_vis/leg {debug_dir}/track_vis/top')
        if debug>=2:
            os.system(f'mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')
        if debug >= 1:
            os.system(f'mkdir -p {debug_dir}/rollouts_vis {debug_dir}/rollouts_ob')
        
        mesh_files = []
        test_scene_dirs = []
        mesh_files.append(f'{code_dir}/FoundationPose/demo_data/square_table_leg/mesh/square_table_leg4.obj')
        mesh_files.append(f'{code_dir}/FoundationPose/demo_data/square_table/mesh/square_table.obj')
        test_scene_dirs.append(f'{code_dir}/FoundationPose/demo_data/square_table_leg')
        test_scene_dirs.append(f'{code_dir}/FoundationPose/demo_data/square_table')
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        meshs = []
        to_origins = []
        extents = []
        bboxs = []
        ests = []
        readers = []
        for i in range(num_poses):
            meshs.append(trimesh.load(mesh_files[i], force='mesh'))
            to_origin, extent = trimesh.bounds.oriented_bounds(meshs[i])
            to_origins.append(to_origin)
            extents.append(extent)
            bboxs.append(np.stack([-extent/2, extent/2], axis=0).reshape(2,3))
            ests.append(FoundationPose(model_pts=meshs[i].vertices, model_normals=meshs[i].vertex_normals, mesh=meshs[i], scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx))
            readers.append(YcbineoatReader(video_dir=test_scene_dirs[i], shorter_side=None, zfar=np.inf))
    
    for i in range(n_rollouts // env.num_envs):
        # Update the progress bar
        pbar.before_round(n_success)

        # Perform a rollout with the current model
        if use_pose_est:
            rollout_data: RolloutSaveValues = rollout(
                env,
                actor,
                rollout_max_steps,
                pbar=pbar,
                resize_video=resize_video,
                n_parts_assemble=n_parts_assemble,
                save_rollouts=save_rollouts,
                ests=ests,
                readers=readers,
                to_origins=to_origins,
                bboxs=bboxs,
                debug=debug,
                debug_dir=debug_dir,
                use_pose_est = use_pose_est,
                num_poses=num_poses,
                iter=i
            )
        else:
            rollout_data: RolloutSaveValues = rollout(
                env,
                actor,
                rollout_max_steps,
                pbar=pbar,
                resize_video=resize_video,
                n_parts_assemble=n_parts_assemble,
                save_rollouts=save_rollouts
            )
        if use_pose_est:
            os.makedirs(f"{debug_dir}/rollouts_ob/leg", exist_ok=True)
            os.makedirs(f"{debug_dir}/rollouts_ob/top", exist_ok=True)
            np.savetxt(f"{debug_dir}/rollouts_ob/leg/leg_poses_{i}.txt", rollout_data.parts_poses.squeeze(0)[:, -14:-7].numpy())
            np.savetxt(f"{debug_dir}/rollouts_ob/top/top_poses_{i}.txt", rollout_data.parts_poses.squeeze(0)[:, :7].numpy())
            # np.savetxt(f"{debug_dir}/parts_poses_{i}.txt", parts_poses.squeeze(0).numpy())
            np.savetxt(f"{debug_dir}/rollouts_ob/leg/leg_poses_est_{i}.txt", rollout_data.parts_poses_est["leg"])
            np.savetxt(f"{debug_dir}/rollouts_ob/top/top_poses_est_{i}.txt", rollout_data.parts_poses_est["top"])
        # Calculate the success rate
        success = rollout_data.rewards.sum(dim=1) == n_parts_assemble
        n_success += success.sum().item()

        # Save the results from the rollout
        if save_rollouts:
            all_robot_states.extend(
                [rollout_data.robot_states[i] for i in range(env.num_envs)]
            )
            all_imgs1.extend(rollout_data.imgs1)
            all_imgs2.extend(rollout_data.imgs2)
            all_actions.extend(rollout_data.actions)
            all_rewards.extend(rollout_data.rewards)
            all_parts_poses.extend(rollout_data.parts_poses)
            all_success.extend(success)

        if break_on_n_success and n_success >= stop_after_n_success:
            print(
                f"Current number of success {n_success} greater than breaking threshold {stop_after_n_success}. Breaking"
            )
            break

    total_reward = np.sum([np.sum(rewards.numpy()) for rewards in all_rewards])
    episode_returns = [
        np.sum(rewards.numpy() * discount ** np.arange(len(rewards)))
        for rewards in all_rewards
    ]

    if record_first_state_only:
        first_robot_states = []
        first_part_poses = []
        first_success = []

    print(f"Checking if we should save rollouts (rollout_save_dir: {rollout_save_dir})")
    if save_rollouts:
        have_img_obs = len(all_imgs1) > 0
        print(
            f"Saving rollouts, have image observations: {have_img_obs} (will make dummy video if False)"
        )
        total_reward = 0
        table_rows = []
        for rollout_idx in trange(
            len(all_robot_states), desc="Saving rollouts", leave=False
        ):
            # Get the rewards and images for this rollout
            robot_states = tensordict_to_list_of_dicts(all_robot_states[rollout_idx])
            actions = all_actions[rollout_idx].numpy()
            rewards = all_rewards[rollout_idx].numpy()
            parts_poses = all_parts_poses[rollout_idx].numpy()
            success = all_success[rollout_idx].item()
            task = env.furniture_name

            if record_first_state_only:
                first_robot_states.append(robot_states[0])
                first_part_poses.append(parts_poses[0])
                first_success.append(success)
                continue

            video1 = (
                all_imgs1[rollout_idx].numpy()
                if have_img_obs
                else np.zeros(
                    (len(robot_states), 2, 2, 3), dtype=np.uint8
                )  # dummy video
            )
            video2 = (
                all_imgs2[rollout_idx].numpy()
                if have_img_obs
                else np.zeros(
                    (len(robot_states), 2, 2, 3), dtype=np.uint8
                )  # dummy video
            )

            # Number of steps until success, i.e., the index of the final reward received
            n_steps = (
                np.where(rewards == 1)[0][-1] + 1 if success else rollout_max_steps
            )

            n_steps += n_steps_padding
            trim_start_steps = 0

            # Stack the two videos side by side into a single video
            # and keep axes as (T, H, W, C) (and cut off after rollout reaches success)
            if have_img_obs:
                video = np.concatenate([video1, video2], axis=2)[
                    trim_start_steps:n_steps
                ]
                video = create_in_memory_mp4(video, fps=20)

            # Calculate the reward and return for this rollout
            episode_return = episode_returns[rollout_idx]

            if save_rollouts_to_wandb and have_img_obs:
                table_rows.append(
                    [
                        wandb.Video(video, fps=20, format="mp4"),
                        success,
                        epoch_idx,
                        np.sum(rewards),
                        episode_return,
                        n_steps,
                    ]
                )

            if rollout_save_dir is not None and (save_failures or success):
                # Save the raw rollout data
                save_raw_rollout(
                    robot_states=robot_states[trim_start_steps : n_steps + 1],
                    imgs1=video1[trim_start_steps : n_steps + 1],
                    imgs2=video2[trim_start_steps : n_steps + 1],
                    parts_poses=parts_poses[trim_start_steps : n_steps + 1],
                    actions=actions[trim_start_steps:n_steps],
                    rewards=rewards[trim_start_steps:n_steps],
                    success=success,
                    task=task,
                    action_type=env.action_type,
                    rollout_save_dir=rollout_save_dir,
                    compress_pickles=compress_pickles,
                )

        if record_first_state_only:
            first_state_npz = str(rollout_save_dir / "first_states.npz")
            print(f"Saving first states to: {first_state_npz}")
            np.savez(
                first_state_npz,
                robot_states=np.asarray(first_robot_states),
                part_poses=np.asarray(first_part_poses),
                success=np.asarray(first_success),
            )

        if save_rollouts_to_wandb:
            # Sort the table rows by return (highest at the top)
            table_rows = sorted(table_rows, key=lambda x: x[4], reverse=True)

            for row in table_rows:
                tbl.add_data(*row)

            # Log the videos to wandb table if a run is active
            if wandb.run is not None:
                wandb.log(
                    {
                        "rollouts": tbl,
                        "epoch": epoch_idx,
                    }
                )

    pbar.close()

    return RolloutStats(
        success_rate=n_success / n_rollouts,
        n_success=n_success,
        n_rollouts=n_rollouts,
        epoch_idx=epoch_idx,
        rollout_max_steps=rollout_max_steps,
        total_return=np.sum(episode_returns),
        total_reward=total_reward,
    )


def do_rollout_evaluation(
    config: DictConfig,
    env: Env,
    save_rollouts_to_file: bool,
    save_rollouts_to_wandb: bool,
    actor: Actor,
    best_success_rate: float,
    epoch_idx: int,
) -> float:
    rollout_save_dir = None

    if save_rollouts_to_file:
        rollout_save_dir = trajectory_save_dir(
            controller=env.ctrl_mode,
            environment="sim",
            task=config.task,
            demo_source="rollout",
            randomness=config.randomness,
            # Don't create here because we have to do it when we save anyway
            create=False,
        )

    actor.set_task(task2idx[config.task])

    rollout_stats = calculate_success_rate(
        env,
        actor,
        n_rollouts=config.rollout.count,
        rollout_max_steps=config.rollout.max_steps,
        epoch_idx=epoch_idx,
        discount=config.discount,
        rollout_save_dir=rollout_save_dir,
        save_rollouts_to_wandb=save_rollouts_to_wandb,
        save_failures=config.rollout.save_failures,
    )
    success_rate = rollout_stats.success_rate
    best_success_rate = max(best_success_rate, success_rate)
    mean_return = rollout_stats.total_return / rollout_stats.n_rollouts

    # Log the success rate to wandb
    wandb.log(
        {
            "success_rate": success_rate,
            "best_success_rate": best_success_rate,
            "epoch_mean_return": mean_return,
            "n_success": rollout_stats.n_success,
            "n_rollouts": rollout_stats.n_rollouts,
            "epoch": epoch_idx,
        }
    )

    return best_success_rate
