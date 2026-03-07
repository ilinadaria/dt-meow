#!/usr/bin/env python3
import rospy
import tf
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import LanePose
from visualization_msgs.msg import Marker, MarkerArray
from dt_robot_utils import get_robot_configuration


class LanePoseVisualizer(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(LanePoseVisualizer, self).__init__(node_name=node_name, node_type=NodeType.DEBUG)

        # Get vehicle name from namespace
        self.veh_name = rospy.get_namespace().strip("/")
        rospy.loginfo(f"[{self.node_name}] Vehicle name: {self.veh_name}")
        self.robot_configuration = get_robot_configuration()

        # Setup publisher
        self.pub_markers = rospy.Publisher(
            "~lane_pose_markers", MarkerArray, queue_size=1, dt_topic_type=TopicType.DEBUG
        )

        # Setup subscriber
        self.sub_lane_pose = rospy.Subscriber("~lane_pose", LanePose, self.cbLanePose, queue_size=1)

        rospy.loginfo(f"[{self.node_name}] Initialzed.")

    def cbLanePose(self, lane_pose_msg):
        marker_array = MarkerArray()
        # rospy.loginfo("[%s] cbLanePose." %(self.node_name))
        marker_array.markers.append(self.lanePose2Marker(lane_pose_msg))
        self.pub_markers.publish(marker_array)

    def lanePose2Marker(self, lane_pose_msg):
        marker = Marker()
        marker.header.frame_id = self.veh_name
        marker.header.stamp = lane_pose_msg.header.stamp
        marker.ns = self.veh_name + "/lane_pose"
        marker.id = 0
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration.from_sec(0.5)
        marker.type = Marker.MESH_RESOURCE
        if self.robot_configuration.name in ("DB18", "DB19", "DB20"):
            marker.mesh_resource = "package://visualization_tools/meshes/DB18.stl"
            marker.color.r = 0.5
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0

        elif self.robot_configuration.name in ("DB21M", "DB21J"):
            marker.mesh_resource = "package://visualization_tools/meshes/DB21J.stl"
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 0.5
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
        else:
            marker.type = Marker.ARROW
            marker.color.r = 0.0
            marker.color.g = 0.5
            marker.color.b = 0.0
            marker.scale.x = 0.3
            marker.scale.y = 0.05
            marker.scale.z = 0.01


        # Get rotation in quaternion
        yaw_quat = tf.transformations.quaternion_about_axis(-lane_pose_msg.phi, [0, 0, 1])
        # rospy.loginfo("[%s] quat: %s "%(self.node_name,yaw_quat))
        marker.pose.orientation.x = yaw_quat[0]
        marker.pose.orientation.y = yaw_quat[1]
        marker.pose.orientation.z = yaw_quat[2]
        marker.pose.orientation.w = yaw_quat[3]

        marker.pose.position.x = 0.0
        marker.pose.position.y = -lane_pose_msg.d
        marker.pose.position.z = 0.0




        if lane_pose_msg.status == LanePose.NORMAL:
            marker.color.a = 1.0
        else:
            marker.color.a = 0.25

        return marker

    def on_shutdown(self):
        rospy.loginfo(f"[{self.node_name}] Shutting down.")


if __name__ == "__main__":
    # Create the NodeName object
    node = LanePoseVisualizer(node_name="lane_pose_visualizer_node")

    # Setup proper shutdown behavior
    rospy.on_shutdown(node.on_shutdown)

    # Keep it spinning to keep the node alive
    rospy.spin()
