#!/usr/bin/env bash
set -e

mkdir data
cd data

wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_odometry.zip
