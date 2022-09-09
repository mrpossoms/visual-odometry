#!/usr/bin/env bash
set -e

if [ -z $VO_DATA_PATH ]; then
echo "Please specify the path for visual odometry datasets:"
read VO_DATA_PATH
fi

# create the directory for the dataset if it doesn't exist
if [ ! -d $VO_DATA_PATH ]; then
	mkdir -p $VO_DATA_PATH
fi

cd $VO_DATA_PATH

function download_and_unzip {
	local URL=$1
	local DATA_ZIP=$(basename $URL)
	local DATA_DIR=$(basename $URL .zip)

	if [ ! -f $DATA_ZIP ] && [ ! -d $DATA_DIR ]; then
		# there is no zip or directory, start from the beginning
		wget --continue $URL
	fi

	if [ -f $DATA_ZIP ]; then
		unzip -n $DATA_ZIP -d $DATA_DIR
	fi
}

# Download datasets and unzip them
download_and_unzip https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip
download_and_unzip https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip
download_and_unzip https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_odometry.zip
download_and_unzip https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip


# Add environment variable to conda environment
cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d

echo "#!/bin/sh" > ./etc/conda/activate.d/env_vars.sh
echo "export VO_DATA_PATH=$VO_DATA_PATH" > ./etc/conda/activate.d/env_vars.sh
echo "#!/bin/sh" > ./etc/conda/deactivate.d/env_vars.sh
echo "unset VO_DATA_PATH" > ./etc/conda/deactivate.d/env_vars.sh

echo "Setup successful, please reactivate your conda environment"

