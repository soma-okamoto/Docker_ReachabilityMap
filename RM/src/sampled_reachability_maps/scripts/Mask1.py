#!/usr/bin/env python
import rospy
from sampled_reachability_maps.srv import Mask_RCNN
from subprocess import Popen
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

def call_mask_client(image_path):
    """
    Mask R-CNNを呼び出して物体検出を行う単体スクリプト。
    :param image_path: 処理する画像のファイルパス
    """
    # ROSノード初期化
    rospy.init_node('mask_client_test', anonymous=True)
    bridge = CvBridge()

    # サーバーノードの起動
    p = Popen(['rosrun', 'sampled_reachability_maps', 'mask_rcnn_server.py'])
    rospy.loginfo('Mask R-CNN server node started')

    # サーバーの準備完了を待機
    rospy.wait_for_service('mask_rcnn')
    rospy.loginfo('Mask R-CNN server is available')

    try:
        # サービスプロキシを作成
        client = rospy.ServiceProxy('mask_rcnn', Mask_RCNN)

        # 画像を読み込んでROSメッセージに変換
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            rospy.logerr(f"画像の読み込みに失敗しました: {image_path}")
            return

        ros_image = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")

        # サービスを呼び出し
        rospy.loginfo("Sending image to Mask R-CNN service...")
        response = client(ros_image)
        rospy.loginfo(f"Detection result: {response}")

        return response
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
    finally:
        # サーバーノードを終了
        p.terminate()
        rospy.loginfo('Mask R-CNN server node terminated')


if __name__ == '__main__':
    # テスト用画像パス（適宜変更してください）
    test_image_path = "/path/to/your/test/image.jpg"
    call_mask_client(test_image_path)

