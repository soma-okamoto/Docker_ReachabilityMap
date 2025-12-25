import os
import argparse
from datetime import datetime
import pickle
import h5py

import math
import numpy as np
from pytorch_kinematics.transforms.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix, euler_angles_to_matrix, matrix_to_euler_angles
import torch

import time
import pdb

# 変数で回転角を指定できるようにする
roll_min, roll_max = -np.pi / 2, np.pi / 2  # ロール角の範囲
pitch_min, pitch_max = -np.pi / 2, np.pi / 2  # ピッチ角の範囲
yaw_min, yaw_max = -np.pi / 2, np.pi / 2  # ヨー角の範囲
# 分割数
num_divisions = 10

# 回転角の範囲を分割
roll_values = np.linspace(roll_min, roll_max, num_divisions)
pitch_values = np.linspace(pitch_min, pitch_max, num_divisions)
yaw_values = np.linspace(yaw_min, yaw_max, num_divisions)

# 2つの対象物に対して処理

# 全ての組み合わせを生成
for roll in roll_values:
    for pitch in pitch_values:
        for yaw in yaw_values:
            # ここでgoal_poseを使って何か処理を行う
            goal_pose_1 = torch.tensor([0.68, -0.73, 0.09, 0.0, 0.0, 0.0])  # 対象物1の位置と姿勢
            goal_pose_2 = torch.tensor([0.50, -0.50, 0.10, 0.0, 0.0, 0.0])  # 対象物2の位置と姿勢（例）

#goal_pose_1 = torch.tensor([0.68, -0.73, 0.09, 0.0, 0.0, 0.0])  # 対象物1の位置と姿勢
#goal_pose_2 = torch.tensor([0.50, -0.50, 0.10, 0.0, 0.0, 0.0])  # 対象物2の位置と姿勢（例）
# goal_poseを動的に作成
#goal_pose = torch.tensor([0.44,-0.7, 0.2, yaw, pitch, roll])  # ここでyaw, pitch, rollを変数で指定
#####16
ang_thresh = np.pi / 6  # threshold for limiting roll and pitch roll angles on the base ground map
dtype = torch.float32  # Choose float32 or 64 etc.
use_vis_freq = True  # Use Visitation frequency as M_score for 3D map (and not Manipulability)

# 6D reachability map settings
angular_res = np.pi / 6  # or 22.5 degrees per bin)
yaw_lim = [-np.pi, np.pi]
pitch_lim = [-np.pi / 2, np.pi / 2]
roll_lim = [-np.pi, np.pi]
yaw_bins = math.ceil((2 * np.pi) / angular_res)  # 16
pitch_bins = math.ceil((np.pi) / angular_res)  # 8. Only half the bins needed (half elevation and full azimuth sufficient to cover sphere)
roll_bins = math.ceil((2 * np.pi) / angular_res)  # 16
cartesian_res = 0.08  # metres 解像度設定
Q_scaling = 200  # Scaling for Q values, adjust this according to your requirement


# Full path and file name to save
parser = argparse.ArgumentParser("create base map")
parser.add_argument("--inv_map_pkl", type=str, required=True, help="")
args, unknown = parser.parse_known_args()

# inv_reach_map_file_path=""
inv_reach_map_file_path = os.path.dirname(args.inv_map_pkl) + '/'
inv_reach_map_file_name = os.path.basename(args.inv_map_pkl)
base_map_file_name = 'base_' + inv_reach_map_file_name

t0 = time.perf_counter()


## Load map
with open(args.inv_map_pkl, 'rb') as f:
    loaded_dict = pickle.load(f)
    inv_transf_batch = loaded_dict['inv_transf_batch']
    M_scores = loaded_dict['M_scores']
    if ("Vis_freq" in loaded_dict.keys()) and use_vis_freq:
        M_scores = loaded_dict['Vis_freq']
        Manip_scaling = 1000000000000000000000000000000
    else:
        Manip_scaling = 1000000000000000000000000000000

# 目標姿勢の変換行列を計 goal_pose1に対しての目標姿勢
goal_transf_1 = torch.zeros((4, 4))
goal_transf_1[:3, :3] = euler_angles_to_matrix(goal_pose_1[3:6], 'ZYX')  # Intrinsic ZYX
goal_transf_1[:, -1] = torch.hstack((goal_pose_1[:3], torch.tensor(1.0)))

#  goal_pose2に対しての目標姿勢
goal_transf_2 = torch.zeros((4, 4))
goal_transf_2[:3, :3] = euler_angles_to_matrix(goal_pose_2[3:6], 'ZYX')  # Intrinsic ZYX
goal_transf_2[:, -1] = torch.hstack((goal_pose_2[:3], torch.tensor(1.0)))

 
# goal_pose1,2に対しての目標姿勢の変換行列をインバースマップのバッチに適用
goal_transf_1 = goal_transf_1.repeat(inv_transf_batch.shape[0], 1, 1)
goal_transf_2 = goal_transf_2.repeat(inv_transf_batch.shape[0], 1, 1)

# 対象物1の位置と姿勢の変換
base_transf_batch_1 = torch.bmm(goal_transf_1, inv_transf_batch)
# 対象物2の位置と姿勢の変換
base_transf_batch_2 = torch.bmm(goal_transf_2, inv_transf_batch)

# 変換されたマップをそれぞれフィルタリング
ground_ind_1 = (base_transf_batch_1[:, 2, 3] >= (-cartesian_res / 2)) & (base_transf_batch_1[:, 2, 3] <= (cartesian_res / 2))
base_transf_batch_1 = base_transf_batch_1[ground_ind_1]
M_scores_1 = M_scores[ground_ind_1]

ground_ind_2 = (base_transf_batch_2[:, 2, 3] >= (-cartesian_res / 2)) & (base_transf_batch_2[:, 2, 3] <= (cartesian_res / 2))
base_transf_batch_2 = base_transf_batch_2[ground_ind_2]
M_scores_2 = M_scores[ground_ind_2]


# 2つの対象物のマップを統合 ここで統合している地面にフィルタリングされたスコアを？？？
base_transf_batch_combined = np.vstack((base_transf_batch_1, base_transf_batch_2))
M_scores_combined = np.hstack((M_scores_1, M_scores_2))
# Debug: Combined batch
print(f"base_transf_batch_combined shape: {base_transf_batch_combined.shape}")


# Ensure base_transf_batch_combined is a Tensor
if isinstance(base_transf_batch_combined, np.ndarray):
    base_transf_batch_combined = torch.tensor(base_transf_batch_combined)

# Ensure euler_angles is a Tensor
if isinstance(euler_angles, np.ndarray):
    euler_angles = torch.tensor(euler_angles)

# Combine 6D poses
base_poses_6d_combined = torch.hstack((
    base_transf_batch_combined[:, :3, 3],
    euler_angles  # Ensure this is a Tensor
))

# Extract rotation matrices and check
rotation_matrices = base_transf_batch_combined[:, :3, :3]
if not isinstance(rotation_matrices, torch.Tensor):
    rotation_matrices = torch.tensor(rotation_matrices, dtype=torch.float32)

# Convert rotation matrices to Euler angles
euler_angles = matrix_to_euler_angles(rotation_matrices, 'ZYX')
print(f"Euler angles shape: {euler_angles.shape}")

# Ensure base_transf_batch_combined is a Tensor
if isinstance(base_transf_batch_combined, np.ndarray):
    base_transf_batch_combined = torch.tensor(base_transf_batch_combined)

# Ensure euler_angles is a Tensor
if isinstance(euler_angles, np.ndarray):
    euler_angles = torch.tensor(euler_angles)

# Combine 6D poses
base_poses_6d_combined = torch.hstack((
    base_transf_batch_combined[:, :3, 3],
    euler_angles  # Ensure this is a Tensor
))


# Filtering
base_poses_6d_combined = torch.hstack((base_transf_batch_combined[:, :3, 3], euler_angles))
filtered_ind = (base_poses_6d_combined[:, 4:6] > -ang_thresh).all(dim=1) & \
               (base_poses_6d_combined[:, 4:6] <= ang_thresh).all(dim=1)

# Apply filter
base_poses_6d_filtered = base_poses_6d_combined[filtered_ind]
base_transf_batch_filtered = base_transf_batch_combined[filtered_ind]
M_scores_filtered = M_scores_combined[filtered_ind]

# Save results
base_poses_6d_filtered = base_poses_6d_filtered.numpy()
base_transf_batch_filtered = base_transf_batch_filtered.numpy()
M_scores_filtered = M_scores_filtered.numpy()

with open(inv_reach_map_file_path + base_map_file_name_combined, 'wb') as f:
    save_dict = {
        "base_poses_6d": base_poses_6d_filtered,
        "M_scores": M_scores_filtered
    }
    pickle.dump(save_dict, f)

print(f"[Saved file: {base_map_file_name_combined}]")

# 保存処理はそのまま（統合後のマップを保存）
base_map_file_name_combined = 'base_combined_' + inv_reach_map_file_name
with open(inv_reach_map_file_path + base_map_file_name_combined, 'wb') as f:
    save_dict = {"base_poses_6d": base_poses_6d, "M_scores": M_scores_combined}
    pickle.dump(save_dict, f)
print(f"[Saved file: {base_map_file_name_combined}]")

# 3D Visualization Map (with numpy)
# 省略部分はそのまま使用

## Create a 3D viz map (with numpy)
# Discretize poses
indices_6d = base_poses_6d / np.array([cartesian_res, cartesian_res, cartesian_res, angular_res, angular_res, angular_res], dtype=np.single)
indices_6d = np.floor(indices_6d)  # Floor to get the appropriate discrete indices
base_poses_6d = indices_6d * np.array([cartesian_res, cartesian_res, cartesian_res, angular_res, angular_res, angular_res], dtype=np.single)
base_poses_6d += np.array([cartesian_res / 2, cartesian_res / 2, cartesian_res / 2, angular_res / 2, angular_res / 2, angular_res / 2], dtype=np.single)

# Loop over all spheres and get 3D sphere array
first = True
for indx in range(base_poses_6d.shape[0]):
    curr_sphere_3d = base_poses_6d[indx, :3]
    if first:
        first = False
        sphere_indxs = (curr_sphere_3d == base_poses_6d[:, :3]).all(axis=1)
        Manip = M_scores[sphere_indxs].mean()
        # Manip = M_scores[sphere_indxs].max() * Manip_scaling # Optional: take max instead of mean

        sphere_array = np.expand_dims(np.append(curr_sphere_3d, Manip), 0)
        pose_array = np.append(base_poses_6d[0, :6], np.array([0., 0., 0., 1.])).astype(np.single)  # dummy value
    else:
        # Check if curr_sphere already exists in the array. If so, skip.
        if ((curr_sphere_3d == sphere_array[:, :3]).all(axis=1).any()):
            continue

        sphere_indxs = (curr_sphere_3d == base_poses_6d[:, :3]).all(axis=1)
        Manip = M_scores[sphere_indxs].mean()
        # Manip = M_scores[sphere_indxs].max() * Manip_scaling # Optional: take max instead of mean

        sphere_array = np.vstack((sphere_array, np.append(curr_sphere_3d, Manip)))
        pose_array = np.vstack((pose_array, np.append(base_poses_6d[indx, :6], np.array([0., 0., 0., 1.])).astype(np.single)))  # dummy value

# Optional: Normalize Q values in the map
min_Q = sphere_array[:, -1].min()
max_Q = sphere_array[:, -1].max()
sphere_array[:, -1] -= min_Q
sphere_array[:, -1] /= (max_Q - min_Q)
sphere_array[:, -1] *= Q_scaling

# Save 3D map as hdf5 (Mimic reuleux data structure)
with h5py.File(inv_reach_map_file_path+"3D_"+base_map_file_name+".h5", 'w') as f:
    sphereGroup = f.create_group('/Spheres')
    sphereDat = sphereGroup.create_dataset('sphere_dataset', data=sphere_array)
    sphereDat.attrs.create('Resolution', data=cartesian_res)
    # (Optional) Save all the 6D poses in each 3D sphere. Currently only dummy pose values (10 dimensional)
    poseGroup = f.create_group('/Poses')
    poseDat = poseGroup.create_dataset('poses_dataset', dtype=float, data=pose_array)
print(f"[Saved file: 3D_{base_map_file_name}.h5 ]")


# END
t_comp = time.perf_counter() - t0
print("[TOTAL Comp Time] = {0:.2e}s".format(t_comp))
