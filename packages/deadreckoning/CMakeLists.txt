#cmake_minimum_required(VERSION 2.8.3)
cmake_minimum_required(VERSION 3.5)

project(deadreckoning)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  # duckietown_msgs # Every duckietown packages should use this.
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
