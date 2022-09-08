#!/usr/bin/env bash
set -e

if [ $VO_DATA_PATH -z ]; then

echo "Please specify the path for visual odometry datasets:"
read VO_DATA_PATH

mkdir -p $VO_DATA_PATH
cd $VO_DATA_PATH

wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_odometry.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip

# Add environment variable to conda environment
cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d

echo "#!/bin/sh" > ./etc/conda/activate.d/env_vars.sh
echo "export VO_DATA_PATH = $VO_DATA_PATH" > ./etc/conda/activate.d/env_vars.sh
echo "#!/bin/sh" > ./etc/conda/deactivate.d/env_vars.sh
echo "unset VO_DATA_PATH" > ./etc/conda/deactivate.d/env_vars.sh

echo "Setup successful, please reactivate your conda environment"
fi

