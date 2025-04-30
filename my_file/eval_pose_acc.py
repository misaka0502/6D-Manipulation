import torch
import numpy as np
import collections
from furniture_bench.config import config
from furniture_bench.utils.pose import get_mat
import furniture_bench.controllers.control_utils as C
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os
ROBOT_HEIGHT = 0.015
table_pos = np.array([0.8, 0.8, 0.4])
table_half_width = 0.015
table_surface_z = table_pos[2] + table_half_width
franka_pose = np.array(
    [0.5 * -table_pos[0] + 0.1, 0, table_surface_z + ROBOT_HEIGHT]
)
base_tag_from_robot_mat = config["robot"]["tag_base_from_robot_base"]
franka_from_origin_mat = get_mat(
    [franka_pose[0], franka_pose[1], franka_pose[2]],
    [0, 0, 0],
)

def sim_to_april_mat():
    return torch.tensor(
        np.linalg.inv(base_tag_from_robot_mat) @ np.linalg.inv(franka_from_origin_mat),
        device="cpu", dtype=torch.float64
    )

def april_to_sim_mat():
        return franka_from_origin_mat @ base_tag_from_robot_mat

def sim_coord_to_april_coord(sim_coord_mat):
    return sim_to_april_mat() @ sim_coord_mat

def april_coord_to_sim_coord(april_coord_mat):
        """Converts AprilTag coordinate to simulator base_tag coordinate."""
        return april_to_sim_mat() @ april_coord_mat

def cam_coord_to_april_coord(pose_est_cam, cam_pos, cam_target):
    cam_pos = np.array(cam_pos)
    cam_target = np.array(cam_target)
    z_camera = (cam_target - cam_pos) / np.linalg.norm(cam_target - cam_pos)
    up_axis = np.array([0, 0, 1])  # Assuming Z is the up axis
    x_camera = -np.cross(up_axis, z_camera)
    x_camera /= np.linalg.norm(x_camera)
    y_camera = np.cross(z_camera, x_camera)
    R_camera_sim = np.vstack([x_camera, y_camera, z_camera]).T
    T_camera_sim = np.eye(4)
    T_camera_sim[:3, :3] = R_camera_sim
    T_camera_sim[:3, 3] = cam_pos
    pos_est_sim = T_camera_sim @ pose_est_cam
    print(pos_est_sim)
    pose_est_april_coord = np.concatenate(
        [
            *C.mat2pose(
                sim_coord_to_april_coord(
                    torch.tensor(pos_est_sim, device="cpu", dtype=torch.float64)
                )
            )
        ]
    )
    print(pose_est_april_coord)
    return pose_est_april_coord

def april_coord_to_cam_coord(pose_est_april, cam_pos, cam_target):
    cam_pos = np.array(cam_pos)
    cam_target = np.array(cam_target)
    z_camera = (cam_target - cam_pos) / np.linalg.norm(cam_target - cam_pos)
    up_axis = np.array([0, 0, 1])  # Assuming Z is the up axis
    x_camera = -np.cross(up_axis, z_camera)
    x_camera /= np.linalg.norm(x_camera)
    y_camera = np.cross(z_camera, x_camera)
    R_camera_sim = np.vstack([x_camera, y_camera, z_camera]).T
    T_camera_sim = np.eye(4)
    T_camera_sim[:3, :3] = R_camera_sim
    T_camera_sim[:3, 3] = cam_pos
    pose_est_april = torch.tensor(pose_est_april, device="cpu", dtype=torch.float64)
    pose_est_april_coord = april_coord_to_sim_coord(
                    C.pose2mat(pose_est_april[:3], pose_est_april[-4:],  device="cpu").numpy()
                )
    pose_est_april_coord = np.linalg.inv(T_camera_sim) @ pose_est_april_coord
    return pose_est_april_coord

# cam_pos = np.array([0.3, -0.65, 0.8])
# cam_target = np.array([0.3, 0.8, 0.00])
# z_camera = (cam_target - cam_pos) / np.linalg.norm(cam_target - cam_pos)
# up_axis = np.array([0, 0, 1])  # Assuming Z is the up axis
# x_camera = -np.cross(up_axis, z_camera)
# x_camera /= np.linalg.norm(x_camera)
# y_camera = np.cross(z_camera, x_camera)
# R_camera_sim = np.vstack([x_camera, y_camera, z_camera]).T
# T_camera_sim = np.eye(4)
# T_camera_sim[:3, :3] = R_camera_sim
# T_camera_sim[:3, 3] = cam_pos
# pose_est = np.array([
#             [ 0.0035802,   0.00353536, -0.9999874 ,  0.19914398],\
#             [ 0.98440486,  0.17586917,  0.00414615,  0.06591379],\
#             [ 0.17588155, -0.9844072 , -0.00285053 , 0.8529349 ],\
#             [ 0.   ,       0.      ,    0.  ,        1.        ]\
#             ])
# pose_est = cam_coord_to_april_coord(pose_est, [0.90, -0.00, 0.65], [-1, -0.00, 0.3])
# print(pose_est)
# pose_est = T_camera_sim @ pose_est
# pose_est = np.concatenate(
#     [
#         *C.mat2pose(
#             sim_coord_to_april_coord(
#                 torch.tensor(pose_est, device="cpu", dtype=torch.float64)
#             )
#         )
#     ]
# )
# pose = np.array([ 0.0048,  0.2422, -0.0157, -0.0319,  0.7064, -0.7064,  0.0319])
# pose_est = np.array([-0.0160879,   0.24492868, -0.01672273, -0.70901746, -0.02294913, 0.01526437, 0.70465213])

# R1 = R.from_quat(pose[-4:]).as_matrix()
# R2 = R.from_quat(pose_est[-4:]).as_matrix()
# R_relative = np.dot(R2, R1.T)
# trace = np.trace(R_relative)
# angle = np.arccos((trace - 1) / 2)
# angle = np.degrees(angle)
# print(angle)
# r = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
# pose_est = C.pose2mat(torch.tensor(pose_est[:3]), torch.tensor(pose_est[-4:]),  device="cpu").numpy()
# pose_est[:3, :3] = pose_est[:3, :3] @ r
# R2 = pose_est[:3, :3]
# R_relative = np.dot(R2, R1.T)
# trace = np.trace(R_relative)
# angle = np.arccos((trace - 1) / 2)
# angle = np.degrees(angle)
# print(angle)


path = "/home2/zxp/Projects/robust-rearrangement/FoundationPose/debug/2025-04-14_11-39-59/rollouts_ob"

error_t_leg = []
error_r_leg = []
t_est_leg = []
t_gt_leg = []
roll_est_leg = []
pitch_est_leg = []
yaw_est_leg = []
roll_gt_leg = []
pitch_gt_leg = []
yaw_gt_leg = []
quat_est_leg = []

error_t_top = []
error_r_top = []
t_est_top = []
t_gt_top = []
roll_est_top = []
pitch_est_top = []
yaw_est_top = []
roll_gt_top = []
pitch_gt_top = []
yaw_gt_top = []
quat_est_top = []

for idx in range(int(len(os.listdir(f"{path}/leg"))/2)):
    error_t_leg = []
    error_r_leg = []
    t_est_leg = []
    t_gt_leg = []
    roll_est_leg = []
    pitch_est_leg = []
    yaw_est_leg = []
    roll_gt_leg = []
    pitch_gt_leg = []
    yaw_gt_leg = []
    quat_est_leg = []
    leg_pose = np.loadtxt(f"{path}/leg/leg_poses_{idx}.txt")
    leg_pose_est = np.loadtxt(f"{path}/leg/leg_poses_est_{idx}.txt")
    for i in range(100):
        error_t_leg.append(np.linalg.norm(leg_pose_est[i][:3] - leg_pose[i][:3]))
        t_est_leg.append(leg_pose_est[i][:3])
        t_gt_leg.append(leg_pose[i][:3])
        rotation_gt = C.pose2mat(torch.tensor(leg_pose[i][:3]), torch.tensor(leg_pose[i][-4:]), device="cpu")[:3, :3].numpy()
        rotation_est = C.pose2mat(torch.tensor(leg_pose_est[i][:3]), torch.tensor(leg_pose_est[i][-4:]), device="cpu")[:3, :3].numpy()
        cos_theta = (np.trace(np.dot(rotation_gt.T, rotation_est)) - 1) / 2
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 避免数值误差
        error_r_leg.append(np.arccos(cos_theta) * 180 / np.pi)  # 转换为角度
        euler_est = R.from_quat(leg_pose_est[i][-4:]).as_euler('xyz', degrees=True)
        euler_gt = R.from_matrix(rotation_gt).as_euler('xyz', degrees=True)
        roll_est_leg.append(euler_est[0])
        pitch_est_leg.append(euler_est[1])
        yaw_est_leg.append(euler_est[2])
        roll_gt_leg.append(euler_gt[0])
        pitch_gt_leg.append(euler_gt[1])
        yaw_gt_leg.append(euler_gt[2])

    te = np.mean(error_t_leg)
    re = np.mean(error_r_leg)
    print(f"idx: {idx}, translation error: {te}")
    print(f"idx: {idx}, rotation error: {re}")
    # fig, axs = plt.subplots(3, 1)
    # t_est_leg = np.array(t_est_leg)
    # t_gt_leg = np.array(t_gt_leg)
    # axs[0].plot(t_est_leg[:, 0], color='r', linewidth=2.0, label="est translation x")
    # axs[0].plot(t_gt_leg[:, 0], color='b', linewidth=1.0, label="gt translation x")
    # axs[1].plot(t_est_leg[:, 1], color='r', linewidth=2.0, label="est translation y")
    # axs[1].plot(t_gt_leg[:, 1], color='b', linewidth=1.0, label="gt translation y")
    # axs[2].plot(t_est_leg[:, 2], color='r', linewidth=2.0, label="est translation z")
    # axs[2].plot(t_gt_leg[:, 2], color='b', linewidth=1.0, label="gt translation z")
    # axs[0].legend()
    # axs[1].legend()
    # axs[2].legend()
    # if leg_pose.shape[0] >= 700:
    #     os.makedirs(f"{path}/failure/tragectory_{idx}/leg/", exist_ok=True)
    #     plt.savefig(f"{path}/failure/tragectory_{idx}/leg/tranlation.png")
    # else:
    #     os.makedirs(f"{path}/success/tragectory_{idx}/leg/", exist_ok=True)
    #     plt.savefig(f"{path}/success/tragectory_{idx}/leg/tranlation.png")

    # roll_est_leg = np.array(roll_est_leg)
    # pitch_est_leg = np.array(pitch_est_leg)
    # yaw_est_leg = np.array(yaw_est_leg)
    # roll_gt_leg = np.array(roll_gt_leg)
    # pitch_gt_leg = np.array(pitch_gt_leg)
    # yaw_gt_leg = np.array(yaw_gt_leg)
    # fig, axs = plt.subplots(3, 1)
    # axs[0].plot(roll_est_leg, color='r', linewidth=2.0, label="roll_est")
    # axs[0].plot(roll_gt_leg, color='b', linewidth=1.0, label="roll_gt")
    # axs[1].plot(pitch_est_leg, color='r', linewidth=2.0, label="pitch_est")
    # axs[1].plot(pitch_gt_leg, color='b', linewidth=1.0, label="pitch_gt")
    # axs[2].plot(yaw_est_leg, color='r', linewidth=2.0, label="yaw_est")
    # axs[2].plot(yaw_gt_leg, color='b', linewidth=1.0, label="yaw_gt")
    # axs[0].legend()
    # axs[1].legend()
    # axs[2].legend()
    # if leg_pose.shape[0] >= 700:
    #     plt.savefig(f"{path}/failure/tragectory_{idx}/leg/rotation.png")
    # else:
    #     plt.savefig(f"{path}/success/tragectory_{idx}/leg/rotation.png")

print("-------------top--------------")
for idx in range(int(len(os.listdir(f"{path}/top"))/2)):
    error_t_top = []
    error_r_top = []
    t_est_top = []
    t_gt_top = []
    roll_est_top = []
    pitch_est_top = []
    yaw_est_top = []
    roll_gt_top = []
    pitch_gt_top = []
    yaw_gt_top = []
    quat_est_top = []
    top_pose = np.loadtxt(f"{path}/top/top_poses_{idx}.txt")
    top_pose_est = np.loadtxt(f"{path}/top/top_poses_est_{idx}.txt")
    for i in range(top_pose.shape[0]):
        error_t_top.append(np.linalg.norm(top_pose_est[i][:3] - top_pose[i][:3]))
        t_est_top.append(top_pose_est[i][:3])
        t_gt_top.append(top_pose[i][:3])
        rotation_gt = C.pose2mat(torch.tensor(top_pose[i][:3]), torch.tensor(top_pose[i][-4:]), device="cpu")[:3, :3].numpy()
        rotation_est = C.pose2mat(torch.tensor(top_pose_est[i][:3]), torch.tensor(top_pose_est[i][-4:]), device="cpu")[:3, :3].numpy()
        cos_theta = (np.trace(np.dot(rotation_gt.T, rotation_est)) - 1) / 2
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 避免数值误差
        error_r_top.append(np.arccos(cos_theta) * 180 / np.pi)  # 转换为角度
        euler_est = R.from_quat(top_pose_est[i][-4:]).as_euler('xyz', degrees=True)
        euler_gt = R.from_matrix(rotation_gt).as_euler('xyz', degrees=True)
        roll_est_top.append(euler_est[0])
        pitch_est_top.append(euler_est[1])
        yaw_est_top.append(euler_est[2])
        roll_gt_top.append(euler_gt[0])
        pitch_gt_top.append(euler_gt[1])
        yaw_gt_top.append(euler_gt[2])

    te = np.mean(error_t_top)
    re = np.mean(error_r_top)
    print(f"idx: {idx}, translation error: {te}")
    print(f"idx: {idx}, rotation error: {re}")
    # t_est_top = np.array(t_est_top)
    # t_gt_top = np.array(t_gt_top)
    # fig, axs = plt.subplots(3, 1)
    # axs[0].plot(t_est_top[:, 0], color='r', linewidth=2.0, label="est translation x")
    # axs[0].plot(t_gt_top[:, 0], color='b', linewidth=1.0, label="gt translation x")
    # axs[1].plot(t_est_top[:, 1], color='r', linewidth=2.0, label="est translation y")
    # axs[1].plot(t_gt_top[:, 1], color='b', linewidth=1.0, label="gt translation y")
    # axs[2].plot(t_est_top[:, 2], color='r', linewidth=2.0, label="est translation z")
    # axs[2].plot(t_gt_top[:, 2], color='b', linewidth=1.0, label="gt translation z")
    # axs[0].legend()
    # axs[1].legend()
    # axs[2].legend()
    # if top_pose.shape[0] >= 700:
    #     os.makedirs(f"{path}/failure/tragectory_{idx}/top/", exist_ok=True)
    #     plt.savefig(f"{path}/failure/tragectory_{idx}/top/tranlation.png")
    # else:
    #     os.makedirs(f"{path}/success/tragectory_{idx}/top/", exist_ok=True)
    #     plt.savefig(f"{path}/success/tragectory_{idx}/top/tranlation.png")

    # roll_est_top = np.array(roll_est_top)
    # pitch_est_top = np.array(pitch_est_top)
    # yaw_est_top = np.array(yaw_est_top)
    # roll_gt_top = np.array(roll_gt_top)
    # pitch_gt_top = np.array(pitch_gt_top)
    # yaw_gt_top = np.array(yaw_gt_top)
    # fig, axs = plt.subplots(3, 1)
    # axs[0].plot(roll_est_top, color='r', linewidth=2.0, label="roll_est")
    # axs[0].plot(roll_gt_top, color='b', linewidth=1.0, label="roll_gt")
    # axs[1].plot(pitch_est_top, color='r', linewidth=2.0, label="pitch_est")
    # axs[1].plot(pitch_gt_top, color='b', linewidth=1.0, label="pitch_gt")
    # axs[2].plot(yaw_est_top, color='r', linewidth=2.0, label="yaw_est")
    # axs[2].plot(yaw_gt_top, color='b', linewidth=1.0, label="yaw_gt")
    # axs[0].legend()
    # axs[1].legend()
    # axs[2].legend()
    # if top_pose.shape[0] >= 700:
    #     plt.savefig(f"{path}/failure/tragectory_{idx}/top/rotation.png")
    # else:
    #     plt.savefig(f"{path}/success/tragectory_{idx}/top/rotation.png")

# leg_pose_foundationpose_path = "/home2/zxp/Projects/Juicer_ws/imitation-juicer/foundationpose/debug/2025-03-02_22-49-35/rollouts_ob/top"
# leg_pose_april_path = "/home2/zxp/Projects/Juicer_ws/imitation-juicer/foundationpose/debug/2025-03-02_22-49-35/top_poses.txt"
# leg_pose_april = np.loadtxt(leg_pose_april_path)
# for i in range(1):
#     leg_pose_foundationpose = np.loadtxt(f"{leg_pose_foundationpose_path}/{i:4d}.txt")
#     leg_pose_cam = april_coord_to_cam_coord(leg_pose_april[i], [0.90, -0.00, 0.65], [-1, -0.00, 0.3])
#     print(leg_pose_foundationpose)
#     print(leg_pose_cam)
# t_est = np.array(t_est)
# t_gt = np.array(t_gt)
# fig, axs = plt.subplots(3, 1)
# axs[0].plot(t_est[:, 0], color='r', linewidth=2.0, label="est translation x")
# axs[0].plot(t_gt[:, 0], color='b', linewidth=1.0, label="gt translation x")
# axs[1].plot(t_est[:, 1], color='r', linewidth=2.0, label="est translation y")
# axs[1].plot(t_gt[:, 1], color='b', linewidth=1.0, label="gt translation y")
# axs[2].plot(t_est[:, 2], color='r', linewidth=2.0, label="est translation z")
# axs[2].plot(t_gt[:, 2], color='b', linewidth=1.0, label="gt translation z")
# axs[0].legend()
# axs[1].legend()
# axs[2].legend()
# plt.show()
# plt.savefig("trajectory_leg-2025-03-02_21-57-45.png")

# roll_est = np.array(roll_est)
# pitch_est = np.array(pitch_est)
# yaw_est = np.array(yaw_est)
# roll_gt = np.array(roll_gt)
# pitch_gt = np.array(pitch_gt)
# yaw_gt = np.array(yaw_gt)
# fig, axs = plt.subplots(3, 1)
# axs[0].plot(roll_est, color='r', linewidth=2.0, label="roll_est")
# axs[0].plot(roll_gt, color='b', linewidth=1.0, label="roll_gt")
# axs[1].plot(pitch_est, color='r', linewidth=2.0, label="pitch_est")
# axs[1].plot(pitch_gt, color='b', linewidth=1.0, label="pitch_gt")
# axs[2].plot(yaw_est, color='r', linewidth=2.0, label="yaw_est")
# axs[2].plot(yaw_gt, color='b', linewidth=1.0, label="yaw_gt")
# axs[0].legend()
# axs[1].legend()
# axs[2].legend()
# plt.show()
# plt.savefig(f"trajectory_{part}_euler.png")
