#!/usr/bin/env python3

import os
import argparse
import pickle
import h5py
import numpy as np
import torch
import time
import rospy
from std_msgs.msg import Float32MultiArray
from pytorch_kinematics.transforms.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_euler_angles
)

import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField


# 回転角の探索範囲と分割数
roll_min, roll_max = -np.pi/2, np.pi/2
pitch_min, pitch_max = -np.pi/2, np.pi/2
yaw_min, yaw_max = -np.pi/2, np.pi/2
num_divisions = 10

# 閾値や解像度設定
ang_thresh = np.pi/6   # 地面マップのロール・ピッチ閾値
angular_res = np.pi/8
cartesian_res = 0.08
Q_scaling = 180

# アーム根本から見たベース原点へのオフセット
ARM_ROOT_OFFSET = torch.tensor([-0.123, 0.0, -0.056], dtype=torch.float32)

def publish_pointcloud(sphere_array, frame_id="base_footprint"):
    # sphere_array は Nx4: [X, Y, Z, score]
    pts = sphere_array.copy()  # Nx4

    # 1) オフセットと軸変換
    offset = np.array([0.123, 0.0, 0.056], dtype=np.float32)
    pts[:, :3] += offset          # ベース→アーム根本変換
    remap = np.empty_like(pts)
    remap[:, 0] = -pts[:, 0]      # X = -X
    remap[:, 1] =  pts[:, 2]      # Y = Z
    remap[:, 2] = -pts[:, 1]      # Z = -Y
    remap[:, 3] =  pts[:, 3]      # score をそのままコピー

    # 2) PointCloud2 メッセージを作成（4フィールド: x,y,z,score）
    header = rospy.Header(stamp=rospy.Time.now(), frame_id=frame_id)
    fields = [
        PointField('x',     0,  PointField.FLOAT32, 1),
        PointField('y',     4,  PointField.FLOAT32, 1),
        PointField('z',     8,  PointField.FLOAT32, 1),
        PointField('score',12,  PointField.FLOAT32, 1),
    ]
    # 各点を (x,y,z,score) のタプルリストに変換
    cloud_data = [tuple(pt) for pt in remap]

    cloud_msg = pc2.create_cloud(header, fields, cloud_data)

    # 3) latched Publisher で配信
    pub = rospy.Publisher("/base_map_pointcloud", PointCloud2, queue_size=1, latch=True)
    rospy.sleep(0.1)
    pub.publish(cloud_msg)
    rospy.loginfo(f"Published PointCloud2 with {len(remap)} points (with score) on /base_map_pointcloud")



def main():
    rospy.init_node('base_map_generator', anonymous=True)
    rospy.loginfo("Waiting for target position on /IRM_Select (Float32MultiArray)...")
    msg = rospy.wait_for_message('/IRM_Select', Float32MultiArray)
    if len(msg.data) < 3:
        rospy.logerr("Received Float32MultiArray with insufficient length (<3)")
        return

    # アーム根本座標系からベース原点へ変換
    raw_xyz = torch.tensor(msg.data[:3], dtype=torch.float32)
    goal_pos = raw_xyz + ARM_ROOT_OFFSET
    x, y, z = goal_pos.tolist()
    rospy.loginfo(f"Received target position (root): {raw_xyz.tolist()}, converted to base: {[x, y, z]}")

    parser = argparse.ArgumentParser(description="Create base reachability map")
    parser.add_argument(
        "--inv_map_pkl", type=str, required=True,
        help="Path to inverse reachability map pickle file"
    )
    args, _ = parser.parse_known_args()

    base_dir = os.path.dirname(args.inv_map_pkl) + '/'
    inv_name = os.path.basename(args.inv_map_pkl)
    base_name = 'base_' + inv_name

    t0 = time.perf_counter()

    # IRM 読み込み
    with open(args.inv_map_pkl, 'rb') as f:
        data = pickle.load(f)
        inv_transf_batch = data['inv_transf_batch']
        M_scores = data.get('Vis_freq', data['M_scores'])

    # 目標姿勢行列生成 (フレーム位置 + 固定姿勢)
    goal_pose = torch.tensor([x, y, z, 0.0, 0.0, 0.0], dtype=torch.float32)
    goal_tf = torch.zeros((4, 4), dtype=torch.float32)
    goal_tf[:3, :3] = euler_angles_to_matrix(goal_pose[3:6], 'ZYX')
    goal_tf[:, -1] = torch.hstack((goal_pose[:3], torch.tensor(1.0)))
    goal_tf = goal_tf.repeat(inv_transf_batch.shape[0], 1, 1)

    # Base map 変換
    base_tf_batch = torch.bmm(goal_tf, inv_transf_batch)

    # 地面上スライス
    ground_idx = (
        (base_tf_batch[:, 2, 3] >= -cartesian_res/2) &
        (base_tf_batch[:, 2, 3] <= cartesian_res/2)
    )
    base_tf_batch = base_tf_batch[ground_idx]
    M_scores = M_scores[ground_idx]

    # ロール・ピッチ制限
    base_poses_6d = torch.hstack((
        base_tf_batch[:, :3, 3],
        matrix_to_euler_angles(base_tf_batch[:, :3, :3], 'ZYX')
    ))
    valid_idx = (
        (base_poses_6d[:, 4:6] > -ang_thresh).all(dim=1) &
        (base_poses_6d[:, 4:6] <= ang_thresh).all(dim=1)
    )
    base_poses_6d = base_poses_6d[valid_idx].numpy()
    M_scores = M_scores[valid_idx].numpy()

    # NumPy pickle 保存
    with open(base_dir + base_name, 'wb') as f:
        pickle.dump({"base_poses_6d": base_poses_6d, "M_scores": M_scores}, f)
    rospy.loginfo(f"[Saved file: {base_name}]")

    # 3D 可視化用 HDF5 保存
    # 座標の離散化
    indices = base_poses_6d / np.array([cartesian_res]*3 + [angular_res]*3, dtype=np.float32)
    indices = np.floor(indices)
    base_poses_6d = indices * np.array([cartesian_res]*3 + [angular_res]*3, dtype=np.float32)
    base_poses_6d += np.array([cartesian_res/2]*3 + [angular_res/2]*3, dtype=np.float32)

    sphere_array = []
    pose_array = []
    seen = {}
    for idx, pose in enumerate(base_poses_6d):
        key = tuple(pose[:3])
        seen.setdefault(key, []).append((pose, M_scores[idx]))
    for key, values in seen.items():
        poses, scores = zip(*values)
        mean_score = float(np.mean(scores))
        sphere_array.append([*key, mean_score])
        first = [*values[0][0], 0.0, 0.0, 0.0, 1.0]
        pose_array.append(first)
    sphere_array = np.array(sphere_array, dtype=np.float32)
    pose_array = np.array(pose_array, dtype=np.float32)

    h5_name = '3D_' + base_name[:-4] + '.h5'
    with h5py.File(base_dir + h5_name, 'w') as hf:
        grp_s = hf.create_group('Spheres')
        grp_s.create_dataset('sphere_dataset', data=sphere_array)
        grp_s['sphere_dataset'].attrs.create('Resolution', data=cartesian_res)
        grp_p = hf.create_group('Poses')
        grp_p.create_dataset('poses_dataset', data=pose_array)
    rospy.loginfo(f"[Saved file: {h5_name}]")

    rospy.loginfo(f"[TOTAL Comp Time] = {time.perf_counter() - t0:.2e}s")

    publish_pointcloud(sphere_array)




if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
