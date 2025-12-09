#cmake_minimum_required(VERSION 2.8.3)
cmake_minimum_required(VERSION 3.5)

project(complete_image_pipeline)

find_package(catkin REQUIRED COMPONENTS
  duckietown_msgs
  rospy
  sensor_msgs
)

catkin_python_setup()

catkin_package()
