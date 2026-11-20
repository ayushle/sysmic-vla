cmake_minimum_required(VERSION 3.8)
project(sysmic_v2_sim)

find_package(ament_cmake REQUIRED)

# Robot description (urdf/: sysmic.urdf + meshes, the SYSMIC 2.0 Fusion export),
# launch file, RViz config, the simulation scripts (sim/ runs on conda Python) and
# the standalone RoboCasa robot (sysmic_v2_robocasa/).
install(
  DIRECTORY urdf launch rviz sim sysmic_v2_robocasa
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "images" EXCLUDE
)

install(
  PROGRAMS scripts/ros2_bridge.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
