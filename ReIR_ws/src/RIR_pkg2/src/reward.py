
# reward.py
import numpy as np

# ハイパーパラメータ
w_a, w_g, w_t = 1.0, 2.0, 1.0
w_s, w_dt      = 0.005, 0.01
w_o, w_f       = 0.5, 1.0       # 新規: orientation, close-fail weights
R_success, R_fail, c_dt = 10.0, 5.0, 0.1

d_close = 0.05  # 近接閾値 [m]

def unpack(obs):
    # obs = [hand_pos(3), hand_vel(3), hand_accel(3),
    #        target_pos(3), ee_pos(3), touch_flag(1), joint_vels(n)]
    hp    =  [0:3]
    hv    = obs[3:6]
    ha    = obs[6:9]
    tp    = obs[9:12]
    ee    = obs[12:15]
    touch = bool(obs[15])
    jv    = obs[16:]
    return hp, hv, ha, tp, ee, touch, jv

# ヘルパー: 把持軸ベクトルを取得（実装例）
def extract_gripper_axis(obs):
    # obsにgripper orientation axisが含まれる場合に取り出す
    # ここではダミーとしてEE→target方向を正規化
    hp, _, _, tp, ee, _, _ = unpack(obs)
    d = tp - ee
    norm = np.linalg.norm(d)
    return d/norm if norm>1e-6 else np.array([1.0,0.0,0.0])


def compute_reward(obs, action, next_obs, done, info):
    # obs: カスタムフォーマットの状態ベクトル
    hp, _, _, tp, ee, touch, _    = unpack(obs)
    hp2, _, _, tp2, ee2, touch2, _ = unpack(next_obs)

    # 1) Alignment
    r_align = np.linalg.norm(ee-hp) - np.linalg.norm(ee2-hp2)
    # 2) Grasping
    r_grasp = np.linalg.norm(ee-tp) - np.linalg.norm(ee2-tp2)
    # 3) Touch
    r_touch = 1.0 if touch2 else 0.0
    # 4) Smoothness & Time
    r_smooth = np.sum(np.square(action))
    r_time   = 1.0

    # 5) Orientation alignment (追加)
    u_vec    = extract_gripper_axis(obs)
    d_vec    = tp - ee
    norm_d   = np.linalg.norm(d_vec)
    r_orient = (u_vec.dot(d_vec) / (norm_d + 1e-6))

    # 6) Close-fail penalty (追加)
    r_fail   = -1.0 if (norm_d <= d_close and not touch2) else 0.0

    # 合算
    r = (w_a * r_align
       + w_g * r_grasp
       + w_t * r_touch
       + w_o * r_orient
       - w_s * r_smooth
       - w_dt * r_time
       + w_f * r_fail)

    # エピソード終了時
    if done:
        if info.get('success', False):
            r += R_success - c_dt * info.get('elapsed_steps', 0)
        else:
            r -= R_fail

    return r

