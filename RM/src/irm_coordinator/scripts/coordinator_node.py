#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
from dataclasses import dataclass
from enum import Enum, auto

from std_msgs.msg import Int32, Float32MultiArray, Bool
from geometry_msgs.msg import Point
from nav_msgs.msg import Path

from irm_coordinator.srv import GenerateBaseMap, GenerateFirstRoute


class State(Enum):
    IDLE = auto()
    WAIT_STABLE = auto()
    GENERATING_BASE = auto()
    GENERATING_ROUTE = auto()
    PUBLISH_ROUTE = auto()
    MOVING = auto()


@dataclass
class Candidate:
    bottle_id: int = -1
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    stamp: rospy.Time = rospy.Time(0)


class CoordinatorNode:
    """
    Coordinator with:
      - stable decision (ID + pose)
      - service calls (base_map, first_route)
      - route publish
      - moving monitoring (route_done)
      - replanning trigger (3)
    """

    def __init__(self):
        rospy.init_node("coordinator_node", anonymous=False)

        # -----------------
        # Parameters
        self.stable_sec = rospy.get_param("~stable_sec", 0.6)       # 安定判定時間
        self.pose_thresh_m = rospy.get_param("~pose_thresh_m", 0.08) # 位置変化閾値
        self.loop_hz = rospy.get_param("~loop_hz", 15)

        # Service names (as you decided)
        self.srv_base_name  = "/generate_base_map"
        self.srv_route_name = "/generate_first_route"

        # Topics
        self.topic_id   = rospy.get_param("~identified_id_topic", "/identified_bottle")
        self.topic_pose = rospy.get_param("~identified_pose_topic", "/identified_bottle_pose")
        self.route_topic = rospy.get_param("~route_topic", "/IRM_first_Route")
        self.route_done_topic = rospy.get_param("~route_done_topic", "/youbot/route_done")

        # BaseMap service request defaults
        self.inv_map_pkl = rospy.get_param("~inv_map_pkl", "")
        self.output_dir  = rospy.get_param("~output_dir", "")
        self.force_regen = rospy.get_param("~force_regenerate", True)

        # FirstRoute service defaults
        self.pc2_topic = rospy.get_param("~pc2_topic", "/IRM_PointCloud2")
        self.frame_id  = rospy.get_param("~frame_id", "base_footprint")
        self.alpha     = float(rospy.get_param("~alpha", 0.5))
        self.spacing   = float(rospy.get_param("~spacing", 0.2))
        self.wait_timeout_sec = float(rospy.get_param("~wait_timeout_sec", 3.0))

        # -----------------
        # State
        self.state = State.IDLE

        self.latest_id = -1
        self.latest_pose = None  # (x,y,z)
        self.latest_stamp = rospy.Time(0)

        self.cand = Candidate()
        self.cand_since = None  # rospy.Time
        self.confirmed = None   # Candidate (stable accepted)
        self.last_plan = None   # Candidate (last planned target)

        self.moving = False
        self.route_done = False

        # -----------------
        # ROS I/O
        self.sub_id = rospy.Subscriber(self.topic_id, Int32, self.on_id, queue_size=10)
        self.sub_pose = rospy.Subscriber(self.topic_pose, Float32MultiArray, self.on_pose, queue_size=10)
        self.sub_done = rospy.Subscriber(self.route_done_topic, Bool, self.on_done, queue_size=10)

        self.pub_route = rospy.Publisher(self.route_topic, Path, queue_size=1, latch=True)

        # Service proxies
        rospy.loginfo("[coordinator_node] waiting services...")
        rospy.wait_for_service(self.srv_base_name)
        rospy.wait_for_service(self.srv_route_name)
        self.srv_base = rospy.ServiceProxy(self.srv_base_name, GenerateBaseMap)
        self.srv_route = rospy.ServiceProxy(self.srv_route_name, GenerateFirstRoute)
        rospy.loginfo("[coordinator_node] services ready.")

        self.rate = rospy.Rate(self.loop_hz)

    def on_id(self, msg: Int32):
        self.latest_id = int(msg.data)
        self.latest_stamp = rospy.Time.now()

    def on_pose(self, msg: Float32MultiArray):
        data = list(msg.data)
        if len(data) < 3:
            return
        self.latest_pose = (float(data[0]), float(data[1]), float(data[2]))
        self.latest_stamp = rospy.Time.now()

    def on_done(self, msg: Bool):
        # baseMove が route 完了時に True を投げる想定
        self.route_done = bool(msg.data)
        if self.route_done:
            self.moving = False

    def _pose_dist(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    def _candidate_changed(self, new_id, new_pose):
        # 初回
        if self.cand_since is None:
            return True

        # ID変化
        if new_id != self.cand.bottle_id:
            return True

        # pose変化（閾値）
        old_pose = (self.cand.x, self.cand.y, self.cand.z)
        if self._pose_dist(new_pose, old_pose) >= self.pose_thresh_m:
            return True

        return False

    def _update_candidate(self):
        if self.latest_pose is None:
            return

        now = rospy.Time.now()
        new_id = self.latest_id
        new_pose = self.latest_pose

        if self._candidate_changed(new_id, new_pose):
            self.cand = Candidate(
                bottle_id=new_id,
                x=new_pose[0], y=new_pose[1], z=new_pose[2],
                stamp=now
            )
            self.cand_since = now
            rospy.loginfo(f"[coordinator_node] candidate update id={new_id} pose=({new_pose[0]:.3f},{new_pose[1]:.3f},{new_pose[2]:.3f})")

    def _is_stable(self):
        if self.cand_since is None:
            return False
        dt = (rospy.Time.now() - self.cand_since).to_sec()
        return dt >= self.stable_sec

    def _needs_replan(self, cand: Candidate):
        """
        Replan condition based on last_plan
        - first time => True
        - ID differs => True
        - pose differs > thresh => True
        """
        if self.last_plan is None:
            return True
        if cand.bottle_id != self.last_plan.bottle_id:
            return True
        dist = self._pose_dist((cand.x,cand.y,cand.z), (self.last_plan.x,self.last_plan.y,self.last_plan.z))
        return dist >= self.pose_thresh_m

    def _call_generate_base(self, cand: Candidate):
        req_target = Point(cand.x, cand.y, cand.z)
        resp = self.srv_base(
            target_pos=req_target,
            inv_map_pkl=self.inv_map_pkl,
            output_dir=self.output_dir,
            force_regenerate=self.force_regen
        )
        if not resp.ok:
            raise RuntimeError(resp.message)
        return resp

    def _call_generate_route(self):
        resp = self.srv_route(
            pc2_topic=self.pc2_topic,
            route_topic="",         # Coordinator が publish するのでここでは publishしない
            frame_id=self.frame_id,
            alpha=self.alpha,
            spacing=self.spacing,
            wait_timeout_sec=self.wait_timeout_sec
        )
        if not resp.ok:
            raise RuntimeError(resp.message)
        return resp

    def spin(self):
        rospy.loginfo("[coordinator_node] start loop.")
        self.state = State.WAIT_STABLE

        while not rospy.is_shutdown():
            try:
                # 1) candidate更新
                self._update_candidate()

                # 2) 状態機械
                if self.state == State.WAIT_STABLE:
                    if self.latest_pose is None:
                        self.rate.sleep()
                        continue

                    if self._is_stable():
                        # stable accepted
                        self.confirmed = self.cand

                        # trigger: ID/pose stable & differs from last plan
                        if self._needs_replan(self.confirmed):
                            # moving中なら「微調整」＝新planで上書き（今回は stop なしで上書き）
                            if self.moving:
                                rospy.logwarn("[coordinator_node] replan triggered while MOVING (micro-adjust).")

                            self.state = State.GENERATING_BASE
                        else:
                            # stableだが同じ => 何もしない
                            pass

                elif self.state == State.GENERATING_BASE:
                    rospy.loginfo(f"[coordinator_node] call generate_base_map id={self.confirmed.bottle_id}")
                    base_resp = self._call_generate_base(self.confirmed)
                    rospy.loginfo(f"[coordinator_node] base_map OK pc2={base_resp.pc2_topic}")
                    self.state = State.GENERATING_ROUTE

                elif self.state == State.GENERATING_ROUTE:
                    rospy.loginfo("[coordinator_node] call generate_first_route")
                    route_resp = self._call_generate_route()
                    self.latest_route = route_resp.route
                    rospy.loginfo(f"[coordinator_node] route OK poses={len(self.latest_route.poses)}")
                    self.state = State.PUBLISH_ROUTE

                elif self.state == State.PUBLISH_ROUTE:
                    # publish route for baseMove
                    self.pub_route.publish(self.latest_route)

                    # update last_plan
                    self.last_plan = self.confirmed

                    # moving start
                    self.moving = True
                    self.route_done = False
                    rospy.loginfo("[coordinator_node] route published -> MOVING")
                    self.state = State.MOVING

                elif self.state == State.MOVING:
                    # moving中も candidate更新 + stable判定で再計画へ
                    if self._is_stable() and self.confirmed is not None:
                        # WAIT_STABLE で confirmed 更新するために一度戻す
                        self.state = State.WAIT_STABLE

                    # 完了したら待機へ
                    if self.route_done:
                        rospy.loginfo("[coordinator_node] route done -> WAIT_STABLE")
                        self.state = State.WAIT_STABLE

                else:
                    self.state = State.WAIT_STABLE

            except Exception as e:
                rospy.logerr(f"[coordinator_node] ERROR in state={self.state}: {e}")
                # エラー時は安全側：一旦待機
                self.state = State.WAIT_STABLE

            self.rate.sleep()


def main():
    node = CoordinatorNode()
    node.spin()

if __name__ == "__main__":
    main()
