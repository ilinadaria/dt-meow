#cmake_minimum_required(VERSION 2.8.3)
cmake_minimum_required(VERSION 3.5)

project(easy_algo)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  #duckietown_msgs # Every duckietown packages should use this.
  #cv_bridge
)


catkin_python_setup()


catkin_package()
 
