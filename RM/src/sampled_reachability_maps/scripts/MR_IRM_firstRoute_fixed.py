#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg    import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg       import Path
from geometry_msgs.msg  import PoseStamped, Point, Quaternion
from std_msgs.msg       import Header

class IRMFirstRouteFromPC2:
    def __init__(self):
        rospy.init_node('irm_route_from_pc2', anonymous=False)

        # パラメータ
        self.pc2_topic    = rospy.get_param('~pc2_topic',    '/IRM_PointCloud2')
        self.route_topic  = rospy.get_param('~route_topic',  '/IRM_first_Route')
        self.frame_id     = rospy.get_param('~frame_id',     'base_footprint')
        self.alpha        = rospy.get_param('~alpha',        0.7)   
        self.spacing      = rospy.get_param('~spacing',      0.4)   # ウェイポイント間隔 [m]
        self.origin       = np.array([0.0, 0.0, 0.0])         # 原点

        # Publisher & Subscriber6
        self.pub = rospy.Publisher(self.route_topic, Path, queue_size=1)
        self.sub = rospy.Subscriber(self.pc2_topic, PointCloud2, self.pc2_callback, queue_size=1)

        rospy.loginfo(f"[IRMFirstRoute] Subscribing PC2: {self.pc2_topic}, Publishing Path: {self.route_topic}")
        rospy.spin()

    def pc2_callback(self, msg: PointCloud2):
        # PointCloud2 から (x,y,z,score) を取り出す
        points = []
        for p in pc2.read_points(msg, field_names=("x","y","z","score"), skip_nans=True):
            points.append(p)
        if not points:
            rospy.logwarn("[IRMFirstRoute] Empty PointCloud2")
            return

        arr = np.array(points)  # shape (N,4)
        coords = arr[:,0:3]
        scores = arr[:,3]

        # 2) スコアを 0～1 に正規化
        max_score = scores.max() if scores.max()>0 else 1.0
        scores_norm = scores / max_score

        # 距離正規化
        dists = np.linalg.norm(coords - self.origin, axis=1)
        maxd  = dists.max() if dists.max()>0 else 1.0
        ndist = dists / maxd
        proximity = 1.0 - ndist


        # 4) 距離重視の重み付け（距離70%, スコア30% など）
        values = self.alpha * proximity + (1.0 - self.alpha) * scores_norm

        idx    = int(np.argmax(values))
        goal   = coords[idx]
        rospy.loginfo(f"[IRMFirstRoute] Goal idx={idx}, coord={goal}, score={scores[idx]:.3f}")

        # 0.5m 間隔で直線上にウェイポイント生成
        vector = goal - self.origin
        dist_total = np.linalg.norm(vector)
        if dist_total < 1e-6:
            rospy.logwarn("[IRMFirstRoute] Goal is too close to origin.")
            waypoints = []
        else:
            direction = vector / dist_total
            # n_steps は 0.5m 間隔で入れられるステップ数
            n_steps = int(dist_total / self.spacing)
            waypoints = [
                self.origin + direction * self.spacing * i
                for i in range(1, n_steps+1)
            ]
            # ゴール位置を必ず末尾に
            waypoints.append(goal)

        # Path メッセージ構築 & Publish
        path = Path()
        now  = rospy.Time.now()
        path.header = Header(stamp=now, frame_id=self.frame_id)
        for wp in waypoints:
            ps = PoseStamped()
            ps.header      = path.header
            ps.pose.position    = Point(x=wp[0], y=wp[1], z=wp[2])
            ps.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
            path.poses.append(ps)

        self.pub.publish(path)
        rospy.loginfo(f"[IRMFirstRoute] Published Path with {len(path.poses)} poses")
        # 一度だけ発行したら解除
        self.sub.unregister()


if __name__=="__main__":
    try:
        IRMFirstRouteFromPC2()
    except rospy.ROSInterruptException:
        pass

        
