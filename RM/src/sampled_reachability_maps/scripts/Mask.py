import rospy
import cv2
import os
import numpy as np
import pickle
from subprocess import Popen, PIPE  # Popen と PIPE を利用する形式
from rgbd_sub.srv import *
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage
import message_filters
import time
from Mask_RCNN.samples import mask_torch
from pathlib import Path

class Gcasp_node:
    def __init__(self):
        rospy.init_node('object_pub', anonymous=True)
        self.bridge = CvBridge()

        parent_dir = str(Path(__file__).resolve().parent)  # どの階層でスクリプトが実行されてもいいように
        self.root_path = parent_dir
        print(self.root_path)

        s = rospy.Service('youbot/search_pick_object_and_grasp_pose', pick_object_search, self.service_f)

        self.depth_sub = message_filters.Subscriber('youbot/camera/aligned_depth_to_color/image_raw', Image)
        self.color_sub = message_filters.Subscriber('youbot/camera/color/image_raw', Image)
        self.mf = message_filters.ApproximateTimeSynchronizer([self.depth_sub, self.color_sub], 100, 0.5)
        self.mf.registerCallback(self.ImageCallback)
        self.sub = False
        self.pub = rospy.Publisher('youbot/object_callback2', Object2Array, queue_size=10)
        self.msg = Object2Array()
        # self.gcasp = gcasp_f.Gcasp()
        self.serv_call = False
        print('inited')

    def get_bbox_center(self, bbox):
        """
        bbox: Bounding box (xmin, ymin, xmax, ymax)
        return: (cx, cy) 物体の中心座標
        """
        xmin, ymin, xmax, ymax = bbox
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        return (cx, cy)

    def mask_client(self, image):
        """
        Mask R-CNNを呼び出し、物体検出を行う
        """
        p = Popen(['rosrun', 'sampled_reachability_maps', 'mask_rcnn_server.py']) 
        print('server node start')
        
        rospy.wait_for_service('mask_rcnn')
        try:
            print('server connect')
            client = rospy.ServiceProxy('mask_rcnn', Mask_RCNN)
            resp1 = client(image)
            return resp1
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def main(self):
        color_path = os.path.join(self.root_path, 'mask/0000_color.png')
        depth_path = os.path.join(self.root_path, 'mask/0000_depth.png')
        cv2.imwrite(color_path, self.color_image)
        cv2.imwrite(depth_path, self.depth_image)

    def ImageCallback(self, color_data, depth_data):
        try:
            self.color_data = color_data
            self.depth_data = depth_data
            self.sub = True
        except Exception as err:
            print(err)
        
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(self.color_data, 'bgr8')
            self.depth_image = self.bridge.imgmsg_to_cv2(self.depth_data, 'passthrough')  # 32FC
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Mask R-CNNで物体検出
        result = self.mask_client(self.color_image)

        # 検出結果が正しく取得できているか確認
        if result:
            # 検出された物体のBBoxの中心座標を計算
            for i, bbox in enumerate(result['rois']):
                # BBoxの中心を求める
                center = self.get_bbox_center(bbox)
                rospy.loginfo(f"Object {i} center: {center}")
                
                # 他の必要な処理をここに追加
	
        # 画像を表示
        cv2.imshow('Color', self.color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("User requested shutdown")

