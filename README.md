just run 
sudo apt install ros-{distribution}-nav2* ros-{distribution}-rtab*


map is at https://drive.google.com/drive/folders/1jJrw21DnUlDW4FFuCJALlUnxAMe1ceHS?usp=drive_link
change directory in slam/src/launch/sysmic_localization.launch.py for the map


launch commands
ros2 launch sysmic_v2_sim sim.launch.py use_sim_time:=true
ros2 launch sysmic_localization.launch.py use_sim_time:=true
ros2 launch sysmic_nav2.launch.py 



