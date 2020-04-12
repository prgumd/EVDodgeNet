#!/usr/bin/bash

FILE_NAME='PRGFlow_Moon.blend'

roslaunch dvs_simulator_py PRGFlow_render.launch
roslaunch dvs_simulator_py PRGFlow_simulate.launch

rosnode kill -a

cd /home/chahatdeep/sim_ws/src/rpg_davis_simulator/datasets/rosbags
BAG_FILE='PRGFlow_Moon_f90-20200319-035255' #Excluding '.bag'
mkdir -p $BAG_FILE\/frames
python ../../scripts/bag_to_images.py --output_dir=$BAG_FILE/frames --image_topic=/dvs/image_raw --bag_file=$BAG_FILE\.bag
rostopic echo -b $BAG_FILE\.bag -p /dvs/pose >> $BAG_FILE\/$BAG_FILE\_pose.csv