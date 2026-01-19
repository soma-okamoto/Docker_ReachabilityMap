#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
import numpy as np

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

import sensor_msgs.point_cloud2 as pc2

from irm_coordinator.srv import GenerateFirstRoute, GenerateFirstRouteResponse


class FirstRouteService:
    """
    Service server: /generate_first_route (irm_coordinator/GenerateFirstRoute)
    - waits PointCloud2 from req.pc2_topic
    - computes a simple first route (your original logic)
    - returns nav_msgs/Path and optionally publishes it if req.route_topic != ""
    """

    def __init__(self):
        rospy.init_node("MR_IRM_firstRoute_service", anonymous=False)

        self.default_pc2_topic   = rospy.get_param("~pc2_topic", "/IRM_PointCloud2")
        self.default_route_topic = rospy.get_param("~route_topic", "/IRM_first_Route")
        self.default_frame_id    = rospy.get_param("~frame_id", "base_footprint")

        self.srv = rospy.Service("/generate_first_route", GenerateFirstRoute, self.handle)

        rospy.loginfo("[MR_IRM_firstRoute_service] ready: /generate_first_route")

    def _pc2_to_points(self, msg: PointCloud2):
        # expects x,y,z,(intensity optional)
        pts = []
        intens = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
            pts.append([p[0], p[1], p[2]])
            intens.append(p[3])
        if len(pts) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return np.array(pts, dtype=np.float32), np.array(intens, dtype=np.float32)

    def _build_path(self, points_xyz, alpha=0.5, spacing=0.2, frame_id="base_footprint"):
        """
        ここはあなたの元コードの firstRoute 生成ロジックを入れる場所。
        いまは「強度が高い点から順に、spacing で間引いて並べる」簡易版。
        """
        path = Path()
        path.header = Header(stamp=rospy.Time.now(), frame_id=frame_id)

        if len(points_xyz) == 0:
            return path

        # 例: x,y 平面で距離間引き
        selected = []
        last = None
        for pt in points_xyz:
            if last is None:
                selected.append(pt)
                last = pt
                continue
            d = np.linalg.norm(pt[:2] - last[:2])
            if d >= spacing:
                selected.append(pt)
                last = pt

        for pt in selected:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(pt[0])
            ps.pose.position.y = float(pt[1])
            ps.pose.position.z = float(pt[2])
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)

        return path

    def handle(self, req):
        res = GenerateFirstRouteResponse()
        try:
            pc2_topic   = req.pc2_topic.strip()   if req.pc2_topic.strip()   else self.default_pc2_topic
            route_topic = req.route_topic.strip() if req.route_topic.strip() else ""
            frame_id    = req.frame_id.strip()    if req.frame_id.strip()    else self.default_frame_id

            timeout = req.wait_timeout_sec if req.wait_timeout_sec > 0 else 3.0

            rospy.loginfo(f"[MR_IRM_firstRoute_service] waiting pc2: {pc2_topic} timeout={timeout}")
            msg = rospy.wait_for_message(pc2_topic, PointCloud2, timeout=timeout)

            pts, intens = self._pc2_to_points(msg)

            # 必要なら強度でソート（例）
            if len(intens) == len(pts) and len(pts) > 0:
                order = np.argsort(-intens)  # high -> low
                pts = pts[order]

            alpha = req.alpha if req.alpha > 0 else 0.5
            spacing = req.spacing if req.spacing > 0 else 0.2

            path = self._build_path(pts, alpha=alpha, spacing=spacing, frame_id=frame_id)

            if route_topic:
                pub = rospy.Publisher(route_topic, Path, queue_size=1, latch=True)
                pub.publish(path)
                rospy.sleep(0.05)

            res.ok = True
            res.message = f"generated route poses={len(path.poses)}"
            res.route = path
            rospy.loginfo(f"[MR_IRM_firstRoute_service] OK {res.message}")
            return res

        except Exception as e:
            res.ok = False
            res.message = f"error: {e}"
            res.route = Path()
            rospy.logerr(f"[MR_IRM_firstRoute_service] {res.message}")
            return res


def main():
    FirstRouteService()
    rospy.spin()

if __name__ == "__main__":
    main()
