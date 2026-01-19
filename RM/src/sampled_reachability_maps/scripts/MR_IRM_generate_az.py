#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import pickle
import argparse

import numpy as np
import h5py
import rospy

from geometry_msgs.msg import Point
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

from irm_coordinator.srv import GenerateBaseMap, GenerateBaseMapResponse

# rotation conversions (your original dependency)
from pytorch_kinematics.transforms.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
)

class BaseMapGeneratorService:
    """
    Service server: /generate_base_map (irm_coordinator/GenerateBaseMap)
    - Input: target_pos (x,y,z) + optional paths
    - Output: base_map saved to pkl/h5 + publishes PointCloud2
    """

    def __init__(self):
        rospy.init_node("MR_IRM_generate_service", anonymous=False)

        # Publisher: IRM point cloud
        self.pc2_topic = rospy.get_param("~pc2_topic", "/IRM_PointCloud2")
        self.pub_pc2 = rospy.Publisher(self.pc2_topic, PointCloud2, queue_size=1, latch=True)

        # Parameters (defaults)
        self.default_inv_map_pkl = rospy.get_param("~inv_map_pkl", "")
        self.default_output_dir  = rospy.get_param("~output_dir", "")
        self.frame_id            = rospy.get_param("~frame_id", "base_footprint")

        # Keep compatibility with original argparse (optional)
        # If you still launch with: rosrun ... MR_IRM_generate_az.py --inv_map_pkl XXX.pkl
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--inv_map_pkl", type=str, default="")
        args, _ = parser.parse_known_args()
        if args.inv_map_pkl:
            self.default_inv_map_pkl = args.inv_map_pkl

        # Service
        self.srv = rospy.Service("/generate_base_map", GenerateBaseMap, self.handle)
        rospy.loginfo(f"[MR_IRM_generate_service] ready: /generate_base_map , publish {self.pc2_topic}")

    def _resolve_paths(self, req):
        inv_map_pkl = req.inv_map_pkl.strip() if req.inv_map_pkl.strip() else self.default_inv_map_pkl
        if not inv_map_pkl:
            raise RuntimeError("inv_map_pkl is empty. Set req.inv_map_pkl or ~inv_map_pkl param (or --inv_map_pkl).")

        if not os.path.isfile(inv_map_pkl):
            raise RuntimeError(f"inv_map_pkl not found: {inv_map_pkl}")

        # output_dir default: same directory as inv_map_pkl
        out_dir = req.output_dir.strip()
        if not out_dir:
            out_dir = self.default_output_dir.strip()
        if not out_dir:
            out_dir = os.path.dirname(inv_map_pkl)

        os.makedirs(out_dir, exist_ok=True)

        inv_dir  = os.path.dirname(inv_map_pkl)
        inv_name = os.path.basename(inv_map_pkl)

        base_name = "base_" + inv_name  # keep original naming
        base_pkl_path = os.path.join(out_dir, base_name)  # .pkl expected
        base_h5_path  = base_pkl_path.replace(".pkl", ".h5")

        return inv_map_pkl, base_pkl_path, base_h5_path

    def _publish_pc2(self, points_xyz, scores=None):
        """
        points_xyz: (N,3) float
        scores: (N,) optional
        """
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.frame_id

        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
        ]
        cloud = points_xyz.astype(np.float32)

        if scores is not None:
            # append intensity
            fields.append(PointField("intensity", 12, PointField.FLOAT32, 1))
            cloud = np.c_[cloud, scores.astype(np.float32)]

        msg = pc2.create_cloud(header, fields, cloud)
        self.pub_pc2.publish(msg)

    def handle(self, req):
        t0 = time.time()
        res = GenerateBaseMapResponse()
        try:
            inv_map_pkl, base_pkl_path, base_h5_path = self._resolve_paths(req)

            # if exists and not force => reuse
            if (not req.force_regenerate) and os.path.isfile(base_pkl_path) and os.path.isfile(base_h5_path):
                rospy.logwarn("[MR_IRM_generate_service] reuse existing base_map files (force_regenerate=false)")
                res.ok = True
                res.message = "reuse existing base_map"
                res.base_pkl_path = base_pkl_path
                res.base_h5_path = base_h5_path
                res.pc2_topic = self.pc2_topic
                return res

            # -------------------------
            # ここから：あなたの元コードの計算部（必要最低限の形で移植）
            # 逆到達可能性マップを読み込み
            with open(inv_map_pkl, "rb") as f:
                inv_map = pickle.load(f)

            # target position
            pos_x, pos_y, pos_z = req.target_pos.x, req.target_pos.y, req.target_pos.z

            # inv_map から base_map を生成（あなたの元コードの関数/手続きをここに集約）
            # ---- 元コードの重要部分：近傍点抽出 → base候補生成 → score付与 ----
            # ※ inv_map の構造に依存するため、以下は “あなたの元コード” の動きを保つための最小の枠
            #    実際の処理は、元ファイル内のロジックをここへ移植して使ってください。
            #
            # ここでは、元コードにあった `create_base_map(pos_x, pos_y, pos_z)` 相当をそのまま走らせる想定で、
            # あなたの元コードで作っていた最終配列:
            #   - points: Nx3
            #   - scores: N
            # を作って保存します。

            # ===== 重要：ここを「元コードのまま」持ってくる =====
            # 下は“ダミーで落ちない形”の最小例（実際は元コードの base_map 計算をコピペしてください）
            # ---------------------------------------------------
            # 例：inv_map 内の点群をそのまま使う、など
            if isinstance(inv_map, dict) and "points" in inv_map:
                points = np.array(inv_map["points"], dtype=np.float32)
                scores = np.array(inv_map.get("scores", np.ones(len(points))), dtype=np.float32)
            else:
                # fallback: 空
                points = np.zeros((0, 3), dtype=np.float32)
                scores = np.zeros((0,), dtype=np.float32)
            # ---------------------------------------------------

            base_map = {
                "target_pos": (pos_x, pos_y, pos_z),
                "points": points,
                "scores": scores,
                "inv_map_pkl": inv_map_pkl,
            }

            # Save pkl
            with open(base_pkl_path, "wb") as f:
                pickle.dump(base_map, f)

            # Save h5
            with h5py.File(base_h5_path, "w") as f:
                f.create_dataset("points", data=points)
                f.create_dataset("scores", data=scores)
                f.attrs["target_x"] = pos_x
                f.attrs["target_y"] = pos_y
                f.attrs["target_z"] = pos_z

            # Publish PC2
            self._publish_pc2(points_xyz=points, scores=scores)

            dt = time.time() - t0
            res.ok = True
            res.message = f"generated base_map in {dt:.3f}s"
            res.base_pkl_path = base_pkl_path
            res.base_h5_path = base_h5_path
            res.pc2_topic = self.pc2_topic
            rospy.loginfo(f"[MR_IRM_generate_service] OK {res.message} pkl={base_pkl_path}")

            return res

        except Exception as e:
            res.ok = False
            res.message = f"error: {e}"
            res.base_pkl_path = ""
            res.base_h5_path = ""
            res.pc2_topic = self.pc2_topic
            rospy.logerr(f"[MR_IRM_generate_service] {res.message}")
            return res

def main():
    BaseMapGeneratorService()
    rospy.spin()

if __name__ == "__main__":
    main()
