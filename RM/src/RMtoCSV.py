import h5py
import numpy as np
import csv

# ファイルパス（必要に応じて変更）
h5file = "/home/dars/RM/src/sampled_reachability_maps/data/3D_base_reach_map_gripper_finger_link_torso_True_0.08_2024-12-20-23-07-48.pklinv_reach_map_gripper_finger_link_torso_True_0.08_2024-12-20-23-07-48.pkl.h5
"
csvfile = "Inverse_reachability_pointcloud.csv"

with h5py.File(h5file, "r") as f:
    spheres = f["/Spheres/sphere_dataset"][:]   # x, y, z, score の配列（N,4）

np.savetxt(csvfile, spheres, delimiter=",")
