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

# 変数で回転角を指定できるようにする
roll_min, roll_max = -np.pi / 2, np.pi / 2  # ロール角の範囲
pitch_min, pitch_max = -np.pi / 2, np.pi / 2  # ピッチ角の範囲
yaw_min, yaw_max = -np.pi / 2, np.pi / 2  # ヨー角の範囲
# 分割数
num_divisions = 1

# 回転角の範囲を分割
roll_values = np.linspace(roll_min, roll_max, num_divisions)
pitch_values = np.linspace(pitch_min, pitch_max, num_divisions)
yaw_values = np.linspace(yaw_min, yaw_max, num_divisions)

# 目標の位置を2つ指定
goal_poses = []
# 1つ目の対象物
for roll in roll_values:
    for pitch in pitch_values:
        for yaw in yaw_values:
            goal_poses.append(torch.tensor([0.68, -0.73, 0.09, yaw, pitch, roll]))  # 1つ目の目標ポーズ追加
# 2つ目の対象物（異なる位置）
for roll in roll_values:
    for pitch in pitch_values:
        for yaw in yaw_values:
            goal_poses.append(torch.tensor([0.74, -0.75, 0.09, yaw, pitch, roll]))  # 2つ目の目標ポーズ追加

# 6D reachability map settings
angular_res = np.pi / 12  # or 22.5 degrees per bin
yaw_lim = [-np.pi, np.pi]
p_lim = [-np.pi / 2, np.pi / 2]
r_lim = [-np.pi, np.pi]
yaw_bins = math.ceil((2 * np.pi) / angular_res)  # 16
pitch_bins = math.ceil((np.pi) / angular_res)  # 8. Only half the bins needed (half elevation and full azimuth sufficient to cover sphere)
roll_bins = math.ceil((2 * np.pi) / angular_res)  # 16
cartesian_res = 0.1  # metres
Q_scaling = 50

# Full path and file name to save
parser = argparse.ArgumentParser("create base map")
parser.add_argument("--inv_map_pkl", type=str, required=True, help="")
args, unknown = parser.parse_known_args()

inv_reach_map_file_path = os.path.dirname(args.inv_map_pkl) + '/'
inv_reach_map_file_name = os.path.basename(args.inv_map_pkl)
base_map_file_name = 'base_' + inv_reach_map_file_name

t0 = time.perf_counter()

## Load map
with open(args.inv_map_pkl, 'rb') as f:
    loaded_dict = pickle.load(f)
    inv_transf_batch = loaded_dict['inv_transf_batch']
    M_scores = loaded_dict['M_scores']
    if ("Vis_freq" in loaded_dict.keys()):
        M_scores = loaded_dict['Vis_freq']
        Manip_scaling = 1000000000000000000000000000000
    else:
        Manip_scaling = 1000000000000000000000000000000

# Base map作成
base_poses_6d_all = []  # 複数の対象物に対してベースポーズを保存するリスト
M_scores_all = []  # 複数の対象物に対するスコアを保存するリスト
base_transf_batch_all = []  # 複数の対象物に対する変換行列を保存するリスト

# 回転角の閾値を設定
ang_thresh = np.pi / 10  # 例として30度の範囲を許容

for goal_pose in goal_poses:
    goal_transf = torch.zeros((4, 4))
    goal_transf[:3, :3] = euler_angles_to_matrix(goal_pose[3:6], 'ZYX')  # Intrinsic ZYX
    goal_transf[:, -1] = torch.hstack((goal_pose[:3], torch.tensor(1.0)))
    goal_transf = goal_transf.repeat(inv_transf_batch.shape[0], 1, 1)
    base_transf_batch = torch.bmm(goal_transf, inv_transf_batch)  # NOTE: broadcasting by yourself and using bmm is much faster than matmul or @!

    ## Slice the base map to only include poses on the ground (z=0)
    # ground_indの作成
# M_scores が tensor 型か確認
    if not isinstance(M_scores, torch.Tensor):
    	print("M_scores is not a tensor. Converting it to tensor.")
    	M_scores = torch.tensor(M_scores)  # M_scores を tensor に変換する
    print(f"M_scores size: {M_scores.size()}")

# base_transf_batchの z 座標（高さ）を取得
    z_coords = base_transf_batch[:, 2, 3]

# ground_indを作成（高さが規定の範囲内であるインデックスを選択）
    ground_ind = (z_coords >= (-cartesian_res / 2)) & (z_coords <= (cartesian_res / 2))

# base_transf_batchの最初の次元とground_indが一致するように調整
# base_transf_batchのサイズとground_indのサイズが一致しない場合
    if ground_ind.size(0) != base_transf_batch.size(0):
    	print("Adjusting ground_ind to match base_transf_batch's size...")
    	# ground_indをbase_transf_batchのサイズに合わせて調整
    	ground_ind = ground_ind[:base_transf_batch.size(0)]

# インデックスを適用
    base_transf_batch = base_transf_batch[ground_ind]



    ## Slice the base map to only include poses with small roll and pitch (within ang_thresh)
    base_poses_6d = torch.hstack((base_transf_batch[:, :3, 3], matrix_to_euler_angles(base_transf_batch[:, :3, :3], 'ZYX')))  # Intrinsic ZYX
    filtered_ind = (base_poses_6d[:, 4:6] > (-ang_thresh)).all(axis=1) & (base_poses_6d[:, 4:6] <= (ang_thresh)).all(axis=1)
    base_poses_6d = base_poses_6d[filtered_ind].numpy()
    base_transf_batch = base_transf_batch[filtered_ind].numpy()
    M_scores = M_scores[filtered_ind].numpy()

    # 複数の対象物の結果をリストに保存
    base_poses_6d_all.append(base_poses_6d)
    M_scores_all.append(M_scores)
    base_transf_batch_all.append(base_transf_batch)

# 結果を保存
base_poses_6d_all = np.vstack(base_poses_6d_all)
M_scores_all = np.vstack(M_scores_all)

with open(inv_reach_map_file_path + base_map_file_name, 'wb') as f:
    save_dict = {"base_poses_6d": base_poses_6d_all, "M_scores": M_scores_all}
    pickle.dump(save_dict, f)
print(f"[Saved file: {base_map_file_name} ]")

# 3D Visualization Map (with numpy)
# 省略部分はそのまま使用

# 3D map作成
first = True
for indx in range(base_poses_6d_all.shape[0]):
    curr_sphere_3d = base_poses_6d_all[indx,:3]
    if first:
        first = False
        sphere_indxs = (curr_sphere_3d == base_poses_6d_all[:,:3]).all(axis=1)
        Manip = M_scores_all[sphere_indxs].mean()
    
        sphere_array = np.expand_dims(np.append(curr_sphere_3d, Manip),0)
        pose_array = np.append(base_poses_6d_all[0,:6], np.array([0., 0., 0., 1.])).astype(np.single) # dummy value
    else:
        if((curr_sphere_3d == sphere_array[:,:3]).all(axis=1).any()):
            continue
        
        sphere_indxs = (curr_sphere_3d == base_poses_6d_all[:,:3]).all(axis=1)
        Manip = M_scores_all[sphere_indxs].mean()
    
        sphere_array = np.vstack((sphere_array, np.append(curr_sphere_3d, Manip)))
        pose_array = np.vstack((pose_array, np.append(base_poses_6d_all[indx,:6], np.array([0., 0., 0., 1.])).astype(np.single))) # dummy value

# # Optional: Normalize Q values in the map
min_Q = sphere_array[:,-1].min()
max_Q = sphere_array[:,-1].max()
sphere_array[:,-1] -= min_Q
sphere_array[:,-1] /= (max_Q-min_Q)
sphere_array[:,-1] *= Q_scaling

# Save 3D map as hdf5 (Mimic reuleux data structure)
with h5py.File(inv_reach_map_file_path+"3D_"+base_map_file_name+".h5", 'w') as f:
    sphereGroup = f.create_group('/Spheres')
    sphereDat = sphereGroup.create_dataset('sphere_dataset', data=sphere_array)
    sphereDat.attrs.create('Resolution', data=cartesian_res)
    
    poseGroup = f.create_group('/Poses')
    poseDat = poseGroup.create_dataset('poses_dataset', dtype=float, data=pose_array)
print(f"[Saved file: 3D_{base_map_file_name}.h5 ]")

# END
t_comp = time.perf_counter() - t0
print("[TOTAL Comp Time] = {0:.2e}s".format(t_comp))

