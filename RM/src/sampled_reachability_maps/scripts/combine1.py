import rospy
import torch
import numpy as np
import pickle
from scipy.spatial import distance_matrix
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import h5py
import math
import matplotlib.pyplot as plt

# 入力ファイルパス
file1 = "/home/dasnote11/catkin_ws/src/sampled_reachability_maps/data/base_inv_reach.pkl"
file2 = "/home/dasnote11/catkin_ws/src/sampled_reachability_maps/data/base_reach_map_gripper_finger_link_torso_True_0.08_2024-12-20-23-07-48.pklinv_reach_map_gripper_finger_link_torso_True_0.08_2024-12-20-23-07-48.pkl"

# 6Dリーチマップ設定
angular_res = np.pi / 4  # 解像度（45度ごと）
yaw_lim = [-np.pi, np.pi]
pitch_lim = [-np.pi / 2, np.pi / 2]
roll_lim = [-np.pi, np.pi]
yaw_bins = math.ceil((2 * np.pi) / angular_res)  # 16
pitch_bins = math.ceil(np.pi / angular_res)  # 8
roll_bins = math.ceil((2 * np.pi) / angular_res)  # 16
cartesian_res = 0.08  # メートル単位
Q_scaling = 20  # スケーリングの調整

# .pklファイルを読み込み
def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data["base_poses_6d"], data["M_scores"]

base_poses_6d_1, M_scores_1 = load_pkl(file1)
base_poses_6d_2, M_scores_2 = load_pkl(file2)

# Torch Tensors に変換
base_poses_6d_1 = torch.tensor(base_poses_6d_1)
base_poses_6d_2 = torch.tensor(base_poses_6d_2)
M_scores_1 = torch.tensor(M_scores_1)
M_scores_2 = torch.tensor(M_scores_2)

# マップの統合
base_poses_6d_combined = torch.vstack((base_poses_6d_1, base_poses_6d_2))
M_scores_combined = torch.hstack((M_scores_1, M_scores_2))

# 距離計算による重なりポイント検出
dist_matrix = distance_matrix(base_poses_6d_combined[:, :3].numpy(), base_poses_6d_combined[:, :3].numpy())
overlap_ind = np.any(dist_matrix < 0.05, axis=1)  # 重なりの閾値を0.05に設定

overlap_points = base_poses_6d_combined[overlap_ind]
overlap_scores = M_scores_combined[overlap_ind]

# M_scoreの高いポイントを抽出（例: 上位20%）
score_threshold = np.percentile(overlap_scores.numpy(), 80)  # 上位20%を抽出
high_score_points = overlap_points[overlap_scores >= score_threshold]
high_score_scores = overlap_scores[overlap_scores >= score_threshold]

# sphere_array と pose_array の作成
sphere_array = high_score_points[:, :3].numpy()  # 高得点の重なりポイントの3D位置
pose_array = high_score_points.numpy()  # 高得点の重なりポイントの6D情報

# Q値の正規化
min_Q = sphere_array[:, -1].min()
max_Q = sphere_array[:, -1].max()
sphere_array[:, -1] -= min_Q
sphere_array[:, -1] /= (max_Q - min_Q)
sphere_array[:, -1] *= Q_scaling

# z座標を0に固定
sphere_array[:, 2] = 0  # z軸座標をすべて0に設定

# ROSノードの初期化とMarkerの準備
def publish_markers(points, scores, frame_id="base_footprint"):
    rospy.init_node('pose_visualizer', anonymous=True)
    marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
    rate = rospy.Rate(10)  # 10Hzで送信

    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "poses"
    marker.id = 0
    marker.type = Marker.SPHERE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.05  # ポイントのサイズ
    marker.scale.y = 0.05
    marker.scale.z = 0.05
    marker.color.a = 1  # 不透明

    # M_scoreを使って色を設定（Qスケーリングを反映）
    min_score = scores.min()
    max_score = scores.max()

    for i, point in enumerate(points):
        p = Point()
        p.x = point[0]
        p.y = point[1]
        p.z = point[2]
        
        # M_scoreに基づく色設定
        normalized_score = (scores[i] - min_score) / (max_score - min_score)  # 正規化
        color = ColorRGBA()
        color.r = 1.0 - normalized_score  # 赤成分
        color.g = normalized_score  # 緑成分
        color.b = 0.0  # 青成分
        color.a = 1.0  # 不透明度

        marker.points.append(p)
        marker.colors.append(color)

    while not rospy.is_shutdown():
        marker_pub.publish(marker)
        rate.sleep()

# 高得点の重なりポイントをrvizで表示
publish_markers(sphere_array, sphere_array[:, -1])

# HDF5ファイルに保存 (高得点の重なり部分)
output_hdf5 = "/home/dasnote11/catkin_ws/src/sampled_reachability_maps/data/high_score_overlap_map.h5"
with h5py.File(output_hdf5, 'w') as f:
    sphereGroup = f.create_group('/Spheres')
    sphereDat = sphereGroup.create_dataset('sphere_dataset', data=sphere_array)
    sphereDat.attrs.create('Resolution', data=cartesian_res)
    poseGroup = f.create_group('/Poses')
    poseDat = poseGroup.create_dataset('poses_dataset', data=pose_array)

print(f"[Saved file: {output_hdf5}]")

# 3D プロットで確認
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(sphere_array[:, 0], sphere_array[:, 1], sphere_array[:, 2], c=sphere_array[:, -1], cmap='jet')
plt.colorbar(scatter)
plt.show()

