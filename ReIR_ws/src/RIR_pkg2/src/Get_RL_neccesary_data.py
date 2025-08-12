#!/usr/bin/env python3
import rospy
import csv
import numpy as np
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Float32MultiArray, Bool,Int32,Float32

class DataLogger:
    def __init__(self):
        # tf2 のセットアップ（base_link→ee_link）
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # MR 手先データ
        self.hand_pos      = None
        self.prev_time     = None
        self.hand_vel      = np.zeros(3)
        self.prev_hand_vel = None
        self.hand_accel    = np.zeros(3)

        # ボトル同定データ
        self.target_pos = None  # [x,y,z]

        # グリッパー接触フラグ
        self.touch_flag = False

        # 関節コマンド
        self.joint_cmd = []

        # CSV ファイル準備
        self.csvfile = open('data_log.csv', 'w', newline='')
        self.writer  = csv.writer(self.csvfile)
        header = [
            'timestamp',
            'hand_x','hand_y','hand_z',
            'hand_vx','hand_vy','hand_vz',
            'hand_ax','hand_ay','hand_az',
            # target position のみ
            'target_x','target_y','target_z',
            'ee_x','ee_y','ee_z',
            'touch_flag',
        ] + [f'joint_cmd_{i}' for i in range(7)]

        self.writer.writerow(header)

        # Subscribers
        rospy.Subscriber('/palm_pose',                Float32MultiArray,   self.cb_hand)
        rospy.Subscriber('/identified_bottle_pose',   Float32MultiArray,   self.cb_target_pos)
        rospy.Subscriber('/arm_joint_command',        Float32MultiArray,   self.cb_joint_cmd)
        rospy.Subscriber('/identified_bottle_touch', Int32, self.cb_touch_flag)

        # 10Hz でログ出力
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_cb)

    def cb_hand(self, msg: PoseStamped):
        t = msg.header.stamp.to_sec() if msg.header.stamp else rospy.get_time()
        pos = np.array([msg.pose.position.x,
                        msg.pose.position.y,
                        msg.pose.position.z], dtype=float)

        if self.hand_pos is not None and self.prev_time is not None:
            dt = t - self.prev_time
            if dt > 1e-6:
                self.hand_vel = (pos - self.hand_pos) / dt
                if self.prev_hand_vel is not None:
                    self.hand_accel = (self.hand_vel - self.prev_hand_vel) / dt
                self.prev_hand_vel = self.hand_vel.copy()

        self.prev_time = t
        self.hand_pos  = pos.copy()
        # rospy.loginfo(pos)

    def cb_target_pos(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=float)
        if data.size == 3:
            self.target_pos = data.copy()
        else:
            rospy.logwarn("Unexpected target pose size: %d", data.size)

    def cb_touch_flag(self, msg: Int32):
        self.touch_flag = int(msg.data)
        # rospy.loginfo(pos)

    def cb_joint_cmd(self, msg: Float32MultiArray):
        self.joint_cmd = list(msg.data)

    def get_ee_position(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'base_link', 'ee_link', rospy.Time(0), rospy.Duration(0.1))
            v = t.transform.translation
            return np.array([v.x, v.y, v.z], dtype=float)
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            return None

    def timer_cb(self, event):
        rospy.loginfo_throttle(2.0, f"hand={self.hand_pos is not None}, "
                                f"target={self.target_pos is not None}, "
                                f"joint_cmd={len(self.joint_cmd)}")

        # TF も確認
        ee_pos = self.get_ee_position()
        if ee_pos is None:
            rospy.logwarn_throttle(5.0, "TF base_link->ee_link not available")

        # 条件が揃ってなくても書く（まずは状況確認したいので）
        ts = event.current_real.to_sec() if event and event.current_real else rospy.get_time()
        row = [
            ts,
            *(self.hand_pos.tolist() if self.hand_pos is not None else [None, None, None]),
            *self.hand_vel.tolist(),
            *self.hand_accel.tolist(),
            *(self.target_pos.tolist() if self.target_pos is not None else [None, None, None]),
            *(ee_pos.tolist() if ee_pos is not None else [None, None, None]),
            int(self.touch_flag)
        ]
        cmds = (self.joint_cmd + [None]*7)[:7]
        row += cmds

        self.writer.writerow(row)
        self.csvfile.flush()

    def shutdown(self):
        self.csvfile.close()



if __name__ == '__main__':
    rospy.init_node('data_logger', anonymous=False)
    logger = DataLogger()
    rospy.on_shutdown(logger.shutdown)
    rospy.spin()
