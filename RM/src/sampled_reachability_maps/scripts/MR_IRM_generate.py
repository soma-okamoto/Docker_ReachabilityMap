#!/usr/bin/env python3
import os
import argparse
import pickle
import h5py
import math
import numpy as np
import torch
import time
import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray, Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2


from pytorch_kinematics.transforms.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
)

class BaseMapGenerator:
    def __init__(self):
        # — 引数パース（一度だけ）
        parser = argparse.ArgumentParser("create base map")
        parser.add_argument("--inv_map_pkl", type=str, required=True,
                            help="入力の逆到達可能性マップ(pickle)ファイル")
        args, unknown = parser.parse_known_args()
        self.inv_path = os.path.dirname(args.inv_map_pkl) + "/"
        self.inv_name = os.path.basename(args.inv_map_pkl)
        self.base_name = "base_" + self.inv_name

        # — 定数・パラメータ（一度だけ）
        self.ang_thresh    = np.pi / 6
        self.dtype         = torch.float32
        self.use_vis_freq  = True
        self.angular_res   = np.pi / 8
        self.cartesian_res = 0.08
        self.Q_scaling     = 180

        # — マップ読み込み（一度だけ）
        with open(self.inv_path + self.inv_name, "rb") as f:
            d = pickle.load(f)
        self.inv_transf_batch = d["inv_transf_batch"]
        if "Vis_freq" in d and self.use_vis_freq:
            self.M_scores = d["Vis_freq"]
        else:
            self.M_scores = d["M_scores"]
        # Manip_scaling は固定値
        self.Manip_scaling = 1e30

        
         # — ROS ノード＆サブスクライバ登録
        rospy.init_node("base_map_generator", anonymous=True)
        self.subscriber = rospy.Subscriber(
            "/IRM_Select",
            Float32MultiArray,
            self.point_callback,
            queue_size=1
        )
         # 出力トピック：点群(x,y,z)＋スコアを [x1,y1,z1,s1, x2,y2,z2,s2, ...] で配信
        self.pub_map = rospy.Publisher("/IRM_Map", Float32MultiArray, queue_size=1)
         # 追加: PointCloud2用 Publisher
        self.pub_map_pc2   = rospy.Publisher("/IRM_PointCloud2", PointCloud2, queue_size=1)


        rospy.loginfo("Waiting for first /IRM_Select message...")
        rospy.spin()

    def point_callback(self, msg:  Float32MultiArray):
        # 最初の一回だけ実行して購読解除
        x, y, z = msg.data[0], msg.data[1], msg.data[2]
        rospy.loginfo(f"[位置受信(1st)] x={x:.3f}, y={y:.3f}, z={z:.3f}")
        self.subscriber.unregister()
        rospy.loginfo("Unsubscribed from /IRM_Select, generating map...")
        self.generate_base_map(torch.tensor([x, y, z], dtype=self.dtype))


    def generate_base_map(self, pos: torch.Tensor):
        # 1) 全姿勢をループ
        roll_vals  = np.linspace(-np.pi/2, np.pi/2, 10)
        pitch_vals = np.linspace(-np.pi/2, np.pi/2, 10)
        yaw_vals   = np.linspace(-np.pi/2, np.pi/2, 10)

        # 2) goal_pose→map変換
        #    （元コードの Transform map by goal_pose ～ Save h5 までをほぼコピペ）
        t0 = time.perf_counter()
        # 2.1）全組み合わせで goal_pose を torch.Tensor に
        goal_poses = []
        for roll in roll_vals:
            for pitch in pitch_vals:
                for yaw in yaw_vals:
                    goal_poses.append(torch.tensor([pos[0], pos[1], pos[2], yaw, pitch, roll], dtype=self.dtype))
        # 2.2）inv_transf_batch と全 goal_pose のバッチ処理
        #     （簡略のため、ここでは最初の goal_pose のみ例示。実際は全てバッチ処理してください）
        goal_pose = goal_poses[0]
        goal_transf = torch.zeros((4, 4), dtype=self.dtype)
        goal_transf[:3, :3] = euler_angles_to_matrix(goal_pose[3:6], "ZYX")
        goal_transf[:, -1]  = torch.hstack((goal_pose[:3], torch.tensor(1.0)))
        goal_transf = goal_transf.repeat(self.inv_transf_batch.shape[0], 1, 1)
        base_transf_batch = torch.bmm(goal_transf, self.inv_transf_batch)

        # 2.3）地面(z≈0)スライス
        mask_ground = (base_transf_batch[:, 2, 3] >= -self.cartesian_res/2) & \
                      (base_transf_batch[:, 2, 3] <=  self.cartesian_res/2)
        base_transf_batch = base_transf_batch[mask_ground]
        M_scores          = self.M_scores[mask_ground]

        # 2.4）ロール・ピッチ閾値スライス
        base_poses_6d = torch.hstack([
            base_transf_batch[:, :3, 3],
            matrix_to_euler_angles(base_transf_batch[:, :3, :3], "ZYX")
        ])
        mask_ang = (base_poses_6d[:, 4:6] > -self.ang_thresh).all(dim=1) & \
                   (base_poses_6d[:, 4:6] <= self.ang_thresh).all(dim=1)
        base_poses_6d     = base_poses_6d[mask_ang].numpy()
        base_transf_batch = base_transf_batch[mask_ang].numpy()
        M_scores          = M_scores[mask_ang].numpy()

        # 2.5）pickle 保存
        with open(self.inv_path + self.base_name, "wb") as f:
            pickle.dump({"base_poses_6d": base_poses_6d, "M_scores": M_scores}, f)
        rospy.loginfo(f"[Saved pickle: {self.base_name}]")

        # 2.6）3Dマップ HDF5 保存（元コード通りに書いてください）
        indices_6d = base_poses_6d / np.array(
            [self.cartesian_res]*3 + [self.angular_res]*3, dtype=np.single
        )
        indices_6d = np.floor(indices_6d)
        base_poses_6d = indices_6d * np.array(
            [self.cartesian_res]*3 + [self.angular_res]*3, dtype=np.single
        )
        base_poses_6d += np.array(
            [self.cartesian_res/2]*3 + [self.angular_res/2]*3, dtype=np.single
        )

        first = True
        sphere_array = None
        pose_array   = None
        for i in range(base_poses_6d.shape[0]):
            curr = base_poses_6d[i, :3]
            if first:
                first = False
                mask = (base_poses_6d[:, :3] == curr).all(axis=1)
                Manip = M_scores[mask].mean()
                sphere_array = np.expand_dims(np.append(curr, Manip), 0)
                pose_array   = np.append(base_poses_6d[0, :6], [0,0,0,1]).astype(np.single)
            else:
                if ((sphere_array[:, :3] == curr).all(axis=1)).any():
                    continue
                mask = (base_poses_6d[:, :3] == curr).all(axis=1)
                Manip = M_scores[mask].mean()
                sphere_array = np.vstack([sphere_array, np.append(curr, Manip)])
                pose_array   = np.vstack([pose_array, np.append(base_poses_6d[i, :6], [0,0,0,1]).astype(np.single)])

        # 正規化＆HDF5書き出し
        min_Q = sphere_array[:, -1].min()
        max_Q = sphere_array[:, -1].max()
        sphere_array[:, -1] = (sphere_array[:, -1] - min_Q) / (max_Q - min_Q) * self.Q_scaling
        with h5py.File(self.inv_path + "3D_" + self.base_name + ".h5", "w") as f:
            g = f.create_group("/Spheres")
            d = g.create_dataset("sphere_dataset", data=sphere_array)
            d.attrs.create("Resolution", data=self.cartesian_res)
            pg = f.create_group("/Poses")
            pg.create_dataset("poses_dataset", data=pose_array)
        rospy.loginfo(f"[Saved HDF5: 3D_{self.base_name}]  time={(time.time()-t0):.2f}s")

         # Unity表示用のためのFloat32MultiArray で配信
        map_msg = Float32MultiArray()
        map_msg.data = sphere_array.flatten().tolist()
        self.pub_map.publish(map_msg)
        rospy.loginfo(f"[IRM_Map published: {sphere_array.shape[0]} points]  time={time.perf_counter()-t0:.2f}s")
        
        # --- 追加: PointCloud2 で配信 ---
        header = Header(stamp=rospy.Time.now(), frame_id="base_footprint")
        fields = [
            PointField('x',      0, PointField.FLOAT32, 1),
            PointField('y',      4, PointField.FLOAT32, 1),
            PointField('z',      8, PointField.FLOAT32, 1),
            PointField('score', 12, PointField.FLOAT32, 1),
        ]
        cloud = pc2.create_cloud(header, fields, sphere_array.tolist())
        self.pub_map_pc2.publish(cloud)
        rospy.loginfo(f"[IRM_PointCloud2 published: {sphere_array.shape[0]} points]  time={time.perf_counter()-t0:.2f}s")



def main():
    try:
        BaseMapGenerator()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()
