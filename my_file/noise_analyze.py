import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.spatial.transform import Rotation as R # 用于处理旋转/四元数
import sys # 用于退出脚本
# font_options = [
#     'WenQuanYi Micro Hei',  # Linux 系统 (如果你在 Linux 上，优先尝试这个)
#     'PingFang SC',         # macOS 系统 (如果你在 macOS 上，优先尝试这个)
#     'SimHei',              # Windows/通用后备 (黑体)
#     'Microsoft YaHei',     # Windows 系统 (微软雅黑，保留以防万一)
#     'Noto Sans CJK SC',    # Linux/通用后备 (思源黑体)
#     'sans-serif'           # 默认后备字体 (无法显示中文)
# ]
# plt.rcParams['font.sans-serif'] = "Noto Serif CJK SC"
# plt.rcParams['axes.unicode_minus'] = False
# --- 用户配置 ---
GROUND_TRUTH_DIR = '/home2/zxp/Projects/Juicer_ws/imitation-juicer/foundationpose/debug/2025-04-09_03-03-29/rollouts_ob/leg/success/gt'  # 存放真值轨迹文件的文件夹路径
ESTIMATED_DIR = '/home2/zxp/Projects/Juicer_ws/imitation-juicer/foundationpose/debug/2025-04-09_03-03-29/rollouts_ob/leg/success/est'      # 存放估计轨迹文件的文件夹路径
FILE_PATTERN = '*.txt'                 # 轨迹文件的格式

# 检查目录是否存在
if not os.path.isdir(GROUND_TRUTH_DIR):
    print(f"错误: 真值目录 '{GROUND_TRUTH_DIR}' 不存在。请创建该目录并放入数据。")
    sys.exit(1)
if not os.path.isdir(ESTIMATED_DIR):
    print(f"错误: 估计目录 '{ESTIMATED_DIR}' 不存在。请创建该目录并放入数据。")
    sys.exit(1)

# --- 数据列配置 ---
# 根据你的描述：前3列是位置(tx,ty,tz)，后4列是四元数(qx,qy,qz,qw)
POS_COLS = [0, 1, 2]
QUAT_COLS = [3, 4, 5, 6] # [x, y, z, w] 顺序

# --- 文件加载配置 ---
# np.loadtxt 的分隔符。对于 np.savetxt 的默认行为，是空格 ' '
DELIMITER = ' '
# --- (如果需要，在此修改 DELIMITER, 例如 DELIMITER = ',') ---

print("--- 配置确认 ---")
print(f"真值数据目录: {GROUND_TRUTH_DIR}")
print(f"估计数据目录: {ESTIMATED_DIR}")
print(f"文件模式: {FILE_PATTERN}")
print(f"位置列索引 (tx, ty, tz): {POS_COLS}")
print(f"四元数列索引 (qx, qy, qz, qw): {QUAT_COLS}")
print(f"使用的列分隔符: '{DELIMITER}' (如果是逗号或其他请修改脚本)")
print("-" * 18 + "\n")


# --- 函数定义 ---

def load_trajectory(filepath, pos_cols, quat_cols, delimiter=None):
    """
    加载单个轨迹文件 (Numpy数组格式的txt文件)。
    返回位置数组 (N, 3) 和四元数数组 (N, 4)。
    """
    if not os.path.exists(filepath):
        print(f"错误: 文件 {filepath} 不存在。")
        return None, None

    try:
        data = np.loadtxt(filepath, delimiter=delimiter)
        if data.size == 0:
             print(f"警告: 文件 {filepath} 为空。跳过。")
             return None, None
        elif data.ndim == 1:
            if len(data) <= max(pos_cols + quat_cols):
                 print(f"警告: 文件 {filepath} 只有一行且列数 ({len(data)}) 不足以提取所需列 (最大索引 {max(pos_cols + quat_cols)})。跳过。")
                 return None, None
            data = data.reshape(1, -1)
        elif data.ndim == 0:
            print(f"警告: 文件 {filepath} 格式异常 (0维数组)。跳过。")
            return None, None

        if data.shape[1] <= max(pos_cols + quat_cols):
            print(f"警告: 文件 {filepath} 的列数 ({data.shape[1]}) 不足以提取所需列 (最大索引 {max(pos_cols + quat_cols)})。跳过。")
            return None, None

        positions = data[:, pos_cols]
        quaternions = data[:, quat_cols]

        norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
        zero_norms = (norms < 1e-10)
        quaternions = np.where(zero_norms, [0.0, 0.0, 0.0, 1.0], quaternions)
        norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        quaternions = quaternions / norms

        # 确认四元数顺序 [x, y, z, w]
        # print("假设输入四元数顺序为 [x, y, z, w]，无需转换。")

        return positions, quaternions
    except ValueError as e:
         print(f"错误: 无法用指定分隔符 '{delimiter}' 解析文件 {filepath}。检查文件内容/DELIMITER。错误: {e}")
         return None, None
    except Exception as e:
        print(f"错误: 加载或处理文件 {filepath} 时发生意外错误: {e}")
        return None, None

def calculate_pose_errors(gt_pos, gt_quat, est_pos, est_quat, gt_filepath, est_filepath):
    """
    计算真值位姿和估计位姿之间的误差。
    返回位置欧氏距离误差 和 方向误差(角度，单位：弧度)。
    """
    if gt_pos is None or gt_quat is None or est_pos is None or est_quat is None:
        return None, None
    if gt_pos.shape[0] == 0 or est_pos.shape[0] == 0:
        print(f"警告: 文件对 {os.path.basename(gt_filepath)} / {os.path.basename(est_filepath)} 中轨迹长度为0。跳过。")
        return None, None
    if gt_pos.shape[1] != 3 or est_pos.shape[1] != 3 or gt_quat.shape[1] != 4 or est_quat.shape[1] != 4:
        print(f"警告: 文件对 {os.path.basename(gt_filepath)} / {os.path.basename(est_filepath)} 数据维度不正确。跳过。")
        return None, None

    min_len = min(len(gt_pos), len(est_pos))
    if len(gt_pos) != len(est_pos):
       print(f"警告: 位姿数量不匹配: {os.path.basename(gt_filepath)} ({len(gt_pos)}) vs {os.path.basename(est_filepath)} ({len(est_pos)}). 截断至 {min_len}。")
       gt_pos = gt_pos[:min_len]
       gt_quat = gt_quat[:min_len]
       est_pos = est_pos[:min_len]
       est_quat = est_quat[:min_len]

    # 1. 位置误差 (仅计算欧氏距离)
    pos_error_vec = est_pos - gt_pos
    pos_error_dist = np.linalg.norm(pos_error_vec, axis=1) # (N,) 的欧氏距离误差

    # 2. 方向误差
    orient_error_angle = np.full(min_len, np.nan) # 初始化为 NaN
    try:
        if np.any(np.isnan(gt_quat)) or np.any(np.isnan(est_quat)):
             print(f"警告: 文件对 {os.path.basename(gt_filepath)} / {os.path.basename(est_filepath)} 含NaN四元数。方向误差为NaN。")
        else:
            r_gt = R.from_quat(gt_quat)
            r_est = R.from_quat(est_quat)
            error_rotation = r_est * r_gt.inv()
            orient_error_angle = error_rotation.magnitude()
    except ValueError as e:
         print(f"错误: 计算方向误差时遇无效四元数: {os.path.basename(gt_filepath)} / {os.path.basename(est_filepath)}: {e}")
    except Exception as e:
        print(f"错误: 计算方向误差时发生意外错误: {os.path.basename(gt_filepath)} / {os.path.basename(est_filepath)}: {e}")

    if np.any(np.isnan(pos_error_dist)):
         print(f"警告: 文件对 {os.path.basename(gt_filepath)} / {os.path.basename(est_filepath)} 位置距离误差含 NaN。")

    # 返回位置距离误差和方向角度误差
    return pos_error_dist, orient_error_angle

# --- 主程序 ---

# 1. 查找轨迹文件
gt_files = sorted(glob.glob(os.path.join(GROUND_TRUTH_DIR, FILE_PATTERN)))
est_files = sorted(glob.glob(os.path.join(ESTIMATED_DIR, FILE_PATTERN)))

# 检查文件数量
if len(gt_files) == 0:
    print(f"错误: 在 '{GROUND_TRUTH_DIR}' 目录下未找到匹配 '{FILE_PATTERN}' 的真值文件。")
    sys.exit(1)
if len(est_files) == 0:
    print(f"错误: 在 '{ESTIMATED_DIR}' 目录下未找到匹配 '{FILE_PATTERN}' 的估计文件。")
    sys.exit(1)

# 比较文件列表长度并配对
if len(gt_files) != len(est_files):
    print(f"警告: 真值文件 ({len(gt_files)}个) 和估计文件 ({len(est_files)}个) 数量不匹配。将处理 {min(len(gt_files), len(est_files))} 对。")
    min_count = min(len(gt_files), len(est_files))
    gt_files = gt_files[:min_count]
    est_files = est_files[:min_count]
else:
    print(f"找到 {len(gt_files)} 对轨迹文件。")
    if len(gt_files) != 15:
         print(f"注意：你提到有30对轨迹，但脚本找到了 {len(gt_files)} 对。")

# 2. 初始化误差列表 (只保留位置距离和方向角度)
all_pos_dist_errors = []
all_orient_angle_errors = [] # 存储角度误差（弧度）
processed_pairs_count = 0
total_poses_processed = 0

# 3. 遍历文件对，计算误差
print("\n--- 开始处理文件对 ---")
for gt_filepath, est_filepath in zip(gt_files, est_files):
    gt_filename = os.path.basename(gt_filepath)
    est_filename = os.path.basename(est_filepath)
    print(f"处理: {gt_filename} <-> {est_filename}")

    gt_pos, gt_quat = load_trajectory(gt_filepath, POS_COLS, QUAT_COLS, delimiter=DELIMITER)
    est_pos, est_quat = load_trajectory(est_filepath, POS_COLS, QUAT_COLS, delimiter=DELIMITER)

    if gt_pos is None or est_pos is None:
        print(f"--> 跳过文件对 (加载失败)。")
        continue

    # 计算误差 (注意返回值的变化)
    pos_dist, orient_angle = calculate_pose_errors(gt_pos, gt_quat, est_pos, est_quat, gt_filepath, est_filepath)

    if pos_dist is not None: # 检查第一个返回值是否成功计算
        num_poses_in_pair = len(pos_dist)
        all_pos_dist_errors.extend(pos_dist)       # 只添加位置距离误差
        all_orient_angle_errors.extend(orient_angle) # 添加方向角度误差 (可能含 NaN)
        processed_pairs_count += 1
        total_poses_processed += num_poses_in_pair
        # print(f"--> 成功计算 {num_poses_in_pair} 个位姿的误差。")
    else:
        print(f"--> 跳过文件对 (误差计算失败)。")

print("--- 文件处理完成 ---")

# 4. 检查是否有有效数据被处理
if processed_pairs_count == 0:
     print("\n错误: 未能成功处理任何文件对。")
     sys.exit(1)
if total_poses_processed == 0:
     print("\n错误: 未能从任何一对文件中提取有效的位姿数据点。")
     sys.exit(1)

print(f"\n成功处理了 {processed_pairs_count} 对文件，共计 {total_poses_processed} 个原始位姿点。")

# 5. 转换为NumPy数组并清理 NaN
all_pos_dist_errors = np.array(all_pos_dist_errors)
all_orient_angle_errors = np.array(all_orient_angle_errors) # 弧度, 可能含 NaN

# 清理位置距离的NaN
nan_mask_pos = np.isnan(all_pos_dist_errors)
num_nan_pos = np.sum(nan_mask_pos)

# 清理方向角度的NaN
nan_mask_orient = np.isnan(all_orient_angle_errors)
num_nan_orient = np.sum(nan_mask_orient)

# 应用掩码，获取有效数据
valid_pos_dist = all_pos_dist_errors[~nan_mask_pos]
valid_orient_angle = all_orient_angle_errors[~nan_mask_orient]

valid_pos_count = len(valid_pos_dist)
valid_orient_count = len(valid_orient_angle)

print("\n--- 数据清理与统计 ---")
if num_nan_pos > 0:
    print(f"警告: 清理了 {num_nan_pos} 个包含NaN的位置距离误差数据点。")
if num_nan_orient > 0:
    print(f"信息: 清理了 {num_nan_orient} 个包含NaN的方向误差数据点。")

print(f"用于统计的有效位置数据点: {valid_pos_count}")
print(f"用于统计的有效方向数据点: {valid_orient_count}")

if valid_pos_count == 0:
     print("错误: 清理NaN后没有有效的位置距离误差数据点。")
     sys.exit(1) # 位置误差是主要的，没有就退出

# 6. 计算统计数据
stats = {}
print("\n位置误差统计 (欧氏距离, 单位: 与输入数据一致):")
stats['pos_dist'] = {'mean': np.mean(valid_pos_dist), 'variance': np.var(valid_pos_dist), 'stddev': np.std(valid_pos_dist)}
print(f"  距离误差: Mean={stats['pos_dist']['mean']:.4f}, Variance={stats['pos_dist']['variance']:.6f}, StdDev={stats['pos_dist']['stddev']:.4f}")

if valid_orient_count > 0:
    print("\n方向误差统计 (角度, 单位: 度):")
    valid_orient_angle_deg = np.degrees(valid_orient_angle) # 转换为度
    stats['orient_angle_deg'] = {'mean': np.mean(valid_orient_angle_deg), 'variance': np.var(valid_orient_angle_deg), 'stddev': np.std(valid_orient_angle_deg)}
    print(f"  角度误差: Mean={stats['orient_angle_deg']['mean']:.4f}, Variance={stats['orient_angle_deg']['variance']:.6f}, StdDev={stats['orient_angle_deg']['stddev']:.4f}")
else:
    print("\n方向误差统计: 无有效数据点，跳过。")
print("-" * 25)

# 7. 绘制直方图
print("\n正在生成直方图...")
# try:
#     plt.style.use('seaborn-v0_8-deep')
# except Exception:
#     print("警告: 未找到 'seaborn-v0_8-deep' 样式，使用默认样式。")
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 
# plt.rcParams['axes.unicode_minus'] = False 
    
# 确定需要多少个图
plot_list = []
plot_list.append(('Position Distance Error (units)', valid_pos_dist))
if valid_orient_count > 0:
    plot_list.append(('Orientation Angle Error (degrees)', valid_orient_angle_deg))

num_plots = len(plot_list)
if num_plots == 0:
    print("没有有效的绘图数据。")

elif num_plots == 1:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    axes_flat = [ax] # 放入列表中以便统一处理
    fig.suptitle('Tranlation Error', fontsize=16)
elif num_plots == 2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6)) # 1行2列
    axes_flat = axes.flatten()
    fig.suptitle('6D Pose Estimation Error Histogram', fontsize=16)
else:
    # 如果未来可能添加更多图，可以扩展布局
    print("警告：绘图数量超出预期，请检查代码。")
    sys.exit(1)


# 绘制每个直方图
for i, (title, data) in enumerate(plot_list):
    ax = axes_flat[i]
    ax.hist(data, bins=50, density=False, alpha=0.75, edgecolor='k', linewidth=0.5)
    ax.set_title(title, fontsize=12)
    if i == 0:
        ax.set_xlabel('Error (meter)', fontsize=10)
    if i == 1:
        ax.set_xlabel('Error (degree)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)

    mean_val = np.mean(data)
    std_val = np.std(data)
    ax.axvline(mean_val, color='r', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.3f}')
    # 可选：添加+/-1标准差线
    # ax.axvline(mean_val + std_val, color='g', linestyle='dotted', linewidth=1, label=f'+1 StdDev')
    # ax.axvline(mean_val - std_val, color='g', linestyle='dotted', linewidth=1, label=f'-1 StdDev')

    ax.legend(fontsize='small')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=9)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# (可选) 保存图形
try:
    fig.savefig("pose_estimation_noise_histograms.svg", format='svg', dpi=300, bbox_inches='tight')
    print("直方图已保存为 pose_estimation_noise_histograms.emf")
except Exception as e:
    print(f"保存图形失败: {e}")

# except Exception as e:
#     print(f"\n错误: 生成或显示绘图时发生错误: {e}")
#     print("请确保已安装 matplotlib (pip install matplotlib)。")

print("\n脚本执行完毕。")