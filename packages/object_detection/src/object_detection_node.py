#!/usr/bin/env python3
import cv2
import numpy as np
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int16 
from nn_model.constants import IMAGE_SIZE
from nn_model.model import Wrapper
from solution.integration_activity import NUMBER_FRAMES_SKIPPED
from duckietown_msgs.msg import (
    AprilTagsWithInfos,
    TagInfo,
    BoolStamped,
    AprilTagDetection,
    AprilTagDetectionArray,
)
from geometry_msgs.msg import Transform, Vector3, Quaternion

class ObjectDetectionNode(DTROS):
    def __init__(self, node_name):
        super(ObjectDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        
        self.initialized = False
        self.frame_id = 0
        
        self.veh = os.environ['VEHICLE_NAME']
        self.pub_vel = rospy.Publisher(f"/{self.veh}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
        self.pub_detections_image = rospy.Publisher("~image/compressed", CompressedImage, queue_size=1)
        self.pub_detection = rospy.Publisher("~sign", Int16, queue_size = 1)
        self.publish_fake_apriltag_detections("~detections", AprilTagDetectionArray, queue_size=1)
        
        
        self.sub_image = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.image_cb,
            buff_size=10000000,
            queue_size=1,
        )


        self.bridge = CvBridge()
        self.model_wrapper = Wrapper(rospy.get_param("~AIDO_eval", False))
        
        self.initialized = True
        self.log("Initialized!")

        # this is cursed
        self.class_to_tag_id = {
            0: 8,   # 4-way-intersect
            1: 11,  # T-intersection
            3: 4,   # no-left-turn
            4: 3,   # no-right-turn
            5: 1    # stop
        }

# this callback should be changed to a different logic (for traffic signs)
    def image_cb(self, image_msg):
        if not self.initialized:
            return

        self.frame_id += 1
        self.frame_id = self.frame_id % (1 + NUMBER_FRAMES_SKIPPED())
        if self.frame_id != 0:
            return

        try:
            bgr = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        except ValueError as e:
            self.logerr(f"Could not decode image: {e}")
            return

        rgb = bgr[..., ::-1] # reverse from bgr to rgb
        rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))

        # bboxes, classes, scores = self.model_wrapper.predict(rgb)
        bboxes, classes, scores = self.model_wrapper.predict_and_filter(rgb)
        for cls, bbox, score in zip(classes, bboxes, scores):
            print(f"cls={cls}, bbox={bbox}, score={score}")


        # Stop logic for ducks and duckiebots
        stop_signal = False
        large_duck = False
        large_duckiebot = False

        # Define the left and right boundaries of the center region
        left_boundary = int(IMAGE_SIZE * 0.33)
        right_boundary = int(IMAGE_SIZE * 0.75)

        for cls, bbox, score in zip(classes, bboxes, scores):

            # Calculate center of bounding box
            center_x = (bbox[0] + bbox[2]) / 2

            if cls == 2 and score > 0.5:  # Duck
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                print("AREA : " + str(area))
                if area > 500 and left_boundary < center_x < right_boundary:
                    stop_signal = True
                    large_duck = True
                    self.log(f"Stop sign")
        

        # Create velocity command
        vel_cmd = Twist2DStamped()
        vel_cmd.header.stamp = rospy.Time.now()
        
        if stop_signal:
            # change state
            vel_cmd.v = 0.0
            vel_cmd.omega = 0.0
            if large_duck and not large_duckiebot:
                self.log("Stopping for Duck.")
            elif large_duckiebot and not large_duck:
                self.log("Stopping for Duckiebot.")
            else:
                self.log("Stopping for duck and duckiebot.")
            
            self.pub_vel.publish(vel_cmd)

            # change state back

        else:
            self.publish_fake_apriltag_detections(image_msg, bboxes, classes, scores)


        self.visualize_detections(rgb, bboxes, classes, scores)

    def visualize_detections(self, rgb, bboxes, classes, scores):
        colors = {0: (0, 255, 255), 1: (0, 165, 255), 2: (255, 0, 0), 3: (0, 255, 0), 4: (255, 0, 255), 5: (255, 255, 0),}
        names = {0: "4-way-intersect", 1: "t-intersection", 2: "duckie", 3: "no-left-turn", 4: "no-right-turn", 5: "stop"}
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for clas, box, score in zip(classes, bboxes, scores):
            pt1 = tuple(map(int, box[:2]))
            pt2 = tuple(map(int, box[2:]))
            color = tuple(reversed(colors[clas])) # opencv expects bgr
            name = names[clas] + " " + str(int(score * 100)) + "%"
            rgb = cv2.rectangle(rgb, pt1, pt2, color, 2)
            text_location = (pt1[0], min(pt2[1] + 30, IMAGE_SIZE))
            rgb = cv2.putText(rgb, name, text_location, font, 1, color, thickness=2)
            
        bgr = rgb[..., ::-1]
        obj_det_img = self.bridge.cv2_to_compressed_imgmsg(bgr)
        self.pub_detections_image.publish(obj_det_img)

    def bbox_to_fake_apriltag_pose(self, bbox):
        # VERY rough transation from bbox to camera pose, since we only use it for random_apriltags node

        x1, y1, x2, y2 = bbox

        w = max(float(x2 - x1), 1.0) # avoid division by 0
        h = max(float(y2 - y1), 1.0)
        area = w * h

        # bbox center
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # normalized image offsets from center
        # these are dimensionless and roughly in [-0.5, 0.5]
        x_offset = (cx - IMAGE_SIZE / 2.0) / IMAGE_SIZE
        y_offset = (cy - IMAGE_SIZE / 2.0) / IMAGE_SIZE

        # crude depth proxy:
        # bigger bbox area => smaller z => closer sign
        z = 5000.0 / area

        # convert image offset into rough camera-frame x/y
        x = x_offset * z
        y = y_offset * z

        return x, y, z
        
    def publish_fake_apriltag_detections(self, image_msg, bboxes, classes, scores):
        det_array = AprilTagDetectionArray()
        det_array.header.stamp = image_msg.header.stamp
        det_array.header.frame_id = image_msg.header.frame_id

        for cls, bbox, score in zip(classes, bboxes, scores):
            cls = int(cls)

            if cls not in self.class_to_tag_id:
                continue

            if score < 0.5:
                continue

            x, y, z = self.bbox_to_fake_apriltag_pose(bbox)

            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            corners = [
                    bbox[0], bbox[1],
                    bbox[2], bbox[1],
                    bbox[2], bbox[3],
                    bbox[0], bbox[3]
                ]
            
            tag_id = self.class_to_tag_id[cls]

            detection = AprilTagDetection(
                transform=Transform(
                    translation=Vector3(x=x, y=y, z=z),
                    rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0), #assume no rotation
                ),
                tag_id=tag_id,
                tag_family="tag36h11",
                hamming=0,
                decision_margin=float(score)*100,
                homography=[0.0]*9,
                center=[cx, cy],
                corners=corners,
                pose_error=0.0
            )


            det_array.detections.append(detection)

        self.pub_detections.publish(det_array)

if __name__ == "__main__":
    object_detection_node = ObjectDetectionNode(node_name="object_detection_node")
    rospy.spin()