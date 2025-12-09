# cmake_minimum_required(VERSION 2.8.3)
cmake_minimum_required(VERSION 3.5)

project(explicit_coordinator)

find_package(catkin REQUIRED COMPONENTS
  duckietown_msgs
  rospy
  std_msgs
)


catkin_python_setup()


catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

