import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import trimesh

# ---- 入力ファイル名
csv_path = 'reachability_pointcloud.csv'

# ---- 1. オフセット値（アーム根本を基準にしたい場合）
offset = np.array([-0.123, 0, -0.056])

# ---- 2. 座標軸変換関数（z-up→y-upへ）
def zup_to_yup(points):
    # x, y, z → x, z, -y
    return np.stack([points[:, 0], points[:, 2], -points[:, 1]], axis=1)

# ---- 点群読み込み
pc = pd.read_csv(csv_path, header=None).values[:, :3]

# ---- オフセット
pc_offset = pc + offset

# ---- 座標変換
pc_final = zup_to_yup(pc_offset)

# ---- 凸包計算（メッシュ化）
hull = ConvexHull(pc_final)
mesh = trimesh.Trimesh(vertices=pc_final, faces=hull.simplices)

# ---- 保存（Unity向けにobj形式がオススメ！）
mesh.export('reachability_hull1.obj')
