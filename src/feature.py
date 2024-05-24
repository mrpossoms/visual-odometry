import cv2
import numpy as np
from .kf import KalmanFilter
from .pinhole import Pinhole
from .utils import *

class Feature3D:
	def __init__(self, camera: Pinhole, keypoints: list[cv2.KeyPoint], poses: list[np.array]):
		assert(len(keypoints) >= 2)
		assert(len(poses) >= 2)

		self._cam = camera
		self._keypoints = keypoints
		self._poses = poses

		# TODO each index into the keypoints collection (-1, or -2) is itself a set of keypoints
		# determine how to handle that case when spawning a new feature instead of just choosing the first
		x0 = point_from_motion(camera, keypoints[-2][0].pt, keypoints[-1][0].pt, poses[-2], poses[-1])
		assert(x0 is not None)

		I = np.eye(3)
		self._kf = KalmanFilter(
			F=I,
			B=I,#np.linalg.inv(poses[-1]),#poses[-1] @ np.linalg.inv(poses[-2]),
			H=I,
			Q=I * 0.1, # proc noise TDB
			R=I, # meas noise TDB
			P=I * 0.1, # state est covar TDB
			x0=x0
			)

	def covariance_ellipse(self) -> tuple[np.array, float, float]:
		# Use the state estimate covariance to project an ellipse into the sensor's xy plane
		# use that ellipse to determine if a keypoint should be considered part of this feature
		centroid = self._cam.project(self._kf.x[:3], None)

		if centroid is None:
			return None, 0, 0

		semi_minor_axes = self._cam.project(self._kf.x[:3] + self._kf.P.diagonal()[:3], None) - centroid
		a = np.linalg.norm(semi_minor_axes[0])
		b = np.linalg.norm(semi_minor_axes[1])

		return centroid, a, b

	def filter_keypoints(self, keypoints: list[cv2.KeyPoint]): #-> Generator[cv2.KeyPoint]:
		# Use the state estimate covariance to project an ellipse into the sensor's xy plane
		# use that ellipse to determine if a keypoint should be considered part of this feature
		centroid, a, b = self.covariance_ellipse()

		if centroid is None:
			return

		a_sqr, b_sqr = a ** 2, b ** 2

		for kp in keypoints:
			delta_sqr = (np.array(kp.pt) - centroid) ** 2
			r = (delta_sqr[0] / a_sqr) + (delta_sqr[1] / b_sqr)

			if r < 1: # then the keypoint is inside the ellipse
				yield kp

	def update(self, keypoints: list[cv2.KeyPoint], pose: np.array):
		# assert(self._kf.x[3] == 1)
		# self._kf.B = np.linalg.inv(pose)
		self._kf.predict(vec(0,0,0)) # we use the control matrix instead of a control 
		# self._kf.x[3] = 1

		my_keypoints = keypoints#self.filter_keypoints(keypoints)

		for kp in keypoints:
			if kp is None or self._keypoints[-1][0] is None:
				continue
			# This _may_ not be needed
			x = point_from_motion(self._cam, kp.pt, self._keypoints[-1][0].pt, pose, self._poses[-1])
			if x is None:
				continue
			# self._kf.H = self._cam.projection(z=x) # Update the projection matrix
			# self._kf.x[3] = 1
			self._kf.update(x)

		self._keypoints += [my_keypoints]

		return self._kf.x[:3]

	def expiring(self) -> bool:
		# Check the magnitude of the state estimate covariance
		return np.linalg.norm(self._kf.P.diagonal()[:3]) > 10

class Feature2D:
	def __init__(self, camera: Pinhole, keypoints: list[cv2.KeyPoint], poses: list[np.array]):
		assert(len(keypoints) >= 2)
		assert(len(poses) >= 2)

		self._cam = camera
		self._keypoints = keypoints
		self._poses = poses

		x, y = keypoints[-1].pt[0], keypoints[-1].pt[1]
		dx, dy = x - keypoints[-2].pt[0], y - keypoints[-2].pt[1]
		x0 = vec(x, y, dx, dy, 0, 0)
		assert(x0 is not None)

		I = np.eye(6)
		self._kf = KalmanFilter(
			F=np.array([
				[1, 0, 1, 0, 0, 0],
				[0, 1, 0, 1, 0, 0],
				[0, 0, 1, 0, 1, 0],
				[0, 0, 0, 1, 0, 1],
				[0, 0, 0, 0, 1, 0],
				[0, 0, 0, 0, 0, 1],
			]),
			B=np.zeros((6,1)),#np.linalg.inv(poses[-1]),#poses[-1] @ np.linalg.inv(poses[-2]),
			H=np.array([
				[1, 0, 0, 0, 0, 0],
				[0, 1, 0, 0, 0, 0],
			]),
			Q=I * 0.1, # proc noise TDB
			R=np.eye(2) * 2, # meas noise TDB
			P=I * 4, # state est covar TDB
			x0=x0
			)

	def covariance_ellipse(self) -> tuple[np.array, float, float]:
		# Use the state estimate covariance to project an ellipse into the sensor's xy plane
		# use that ellipse to determine if a keypoint should be considered part of this feature
		centroid = self._kf.x[:2]
		semi_minor_axes = self._kf.P.diagonal()[:2]
		a = semi_minor_axes[0]
		b = semi_minor_axes[1]

		return centroid, a, b

	def filter_keypoints(self, keypoint: cv2.KeyPoint): #-> Generator[cv2.KeyPoint]:
		# Use the state estimate covariance to project an ellipse into the sensor's xy plane
		# use that ellipse to determine if a keypoint should be considered part of this feature
		centroid, a, b = self.covariance_ellipse()

		if centroid is None:
			return

		a_sqr, b_sqr = a ** 2, b ** 2


		delta_sqr = (np.array(keypoint.pt) - centroid) ** 2
		r = (delta_sqr[0] / a_sqr) + (delta_sqr[1] / b_sqr)

		if r < 2: # then the keypoint is inside the ellipse
			yield keypoint

	def update(self, keypoint: cv2.KeyPoint, pose: np.array):
		
		self._kf.predict(vec(0)) 
		# self._kf.x[3] = 1

		# my_keypoints = self.filter_keypoints(keypoints)

		if keypoint is not None:
			# self._kf.H = self._cam.projection(z=x) # Update the projection matrix
			# self._kf.x[3] = 1
			self._kf.update(keypoint.pt)

		self._keypoints.append(keypoint)

		return self._kf.x

	def expiring(self) -> bool:
		# Check the magnitude of the state estimate covariance
		return np.linalg.norm(self._kf.P.diagonal()[:2]) > 10

class DummyKp:
	def __init__(self, pt):
		self.pt = pt

if __name__ == "__main__":
	import os
	import argparse
	from PIL import Image
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument("--output", type=str, default="motion.gif")
	arg_parser.add_argument("--frames", type=int, default=25)
	args = arg_parser.parse_args()
		
	I = np.zeros((args.frames, 128, 128, 3), dtype=np.uint8) * 255
	keypoints = []
	poses = []
	recovered_points = []
	dp = vec(0, 0, 0.5)
	cam = Pinhole(I[0])
	feat = None
	
	p = vec(1, 1, 1) + vec(0, 0, 10)

	for f in range(args.frames):
		cp = dp * (f+4) # camera pos
		T = translate(cp)
		poses.append(T)
		cam.set_frame(I[f])
		cam.set_transform(T)

		z = cam.project(p, np.array([255] * 3, dtype=np.uint8))
		if z is not None:
			keypoints.append([DummyKp(z)])
		else:
			keypoints.append([None])

		if f > 1 and feat is None:
			feat = Feature3D(cam, keypoints[-2:], poses[-2:])
		
		if feat is not None:
			p_prime = feat.update(keypoints[-1], poses[-1])
			print(p_prime)
			coord = cam.project(feat._kf.x[:3], np.array([255, 0, 0], dtype=np.uint8))
			if coord is not None:
				centroid, a, b = feat.covariance_ellipse()
				cv2.ellipse(I[f], (int(centroid[0]), int(centroid[1])), (int(a), int(b)), 0, 0, 360, (0, 255, 0), 1)
				
			if feat.expiring():
				feat = None
				print("Feature expired")


	imgs = [Image.fromarray(
		img).resize((256, 256), resample=Image.NEAREST) for img in I]
	imgs[0].save("feature.gif", save_all=True, append_images=imgs[1:], duration=10, loop=0)

	os.system("imgcat feature.gif")