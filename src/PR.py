import cv2
import numpy as np
from src.kf import KalmanFilter
from src.feature import Feature2D
from src.pinhole import Pinhole
from src.utils import *

class PR:
	class State:
		def __init__(self, I: np.array, camera: Pinhole, pose: np.array, features: list[Feature2D]):
			self.I = I
			self.camera = camera
			self.pose = pose
			self.features = features
			# computed later
			self.keypoints = None
			self.descriptors = None

		def point_estimates(self) -> list[np.array]:
			return [f._point_estimate for f in self.features]

		def detect_keypoints(self, detector):
			self.keypoints, self.descriptors = detector.detectAndCompute(self.I, None)

		def update_features(self, matcher: cv2.BFMatcher, x_t_1) -> list[Feature2D]:
			assert(self.keypoints is not None)

			features_t1 = []

			keypoint_matches = [-1] * len(self.keypoints)

			# matches = self.matcher.match(self.descriptors, x_t_1.descriptors) # order of params is queryIdx, trainIdx
			# feature_matches = [-1] * len(matches)  

			# We need to do two things here
			# 1. update existing features
			# 2. spawn new features
			for fi, feature in enumerate(self.features):
				feature.predict()

				# Option 1, use a matcher to match keypoints explicitly
				# for mi, match in enumerate(feature_matches):
				# 	kp = self.keypoints[match.queryIdx]
				# 	if feature.candidate_keypoint(kp):
				# 		feature_matches[mi] = fi
				# 		feature.update(kp, self.pose)
				
				# Option 2, update on all keypoints that qualify as candidates
				for ki, kp in enumerate(self.keypoints):
					if feature.candidate_keypoint(kp):
						feature.update(kp, self.pose)
						keypoint_matches[ki] = fi

				if not feature.expiring():
					features_t1.append(feature)
				else:
					print("expiring")

			for ki, fi in enumerate(keypoint_matches): # identify features that didn't have matches
				if fi == -1:
					features_t1.append(Feature2D(self.camera, [self.keypoints[ki]], [self.pose]))

			return features_t1

	def __init__(self, camera):
		self.camera = camera
		self.X = []

		self.orb = cv2.ORB_create(patchSize=5, nlevels=1)
		self.matcher = cv2.BFMatcher() #cv2.NORM_HAMMING, crossCheck=True)


	def reset(self):
		self.X = []

	def append(self, frame, cam_pose, max_distance=None) -> list[np.array]:
		x_t_1 = self.X[-1] if len(self.X) > 0 else None
		x_t = PR.State(frame, self.camera, cam_pose, x_t_1.features if x_t_1 is not None else [])

		x_t.detect_keypoints(self.orb)
		x_t.features = x_t.update_features(self.matcher, x_t_1)

		self.X.append(x_t)

		return self.X[-1].point_estimates()


if __name__ == "__main__":
	import argparse
	from PIL import Image
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument("--output", type=str, default="motion1.gif")
	arg_parser.add_argument("--frames", type=int, default=25)
	args = arg_parser.parse_args()
	
	I = np.zeros((args.frames, 128, 128, 3), dtype=np.uint8)

	# verts = [vec(1,1,1) + vec(0, 0, 5), vec(-1,2,1) + vec(0, 0, 10)]
	verts, colors = cube(d=2, position=vec(0, 0, 7))
	verts = verts[:5]

	dp = vec(0, 0, 0.5)
	cam = Pinhole(I[0])
	pr = PR(cam)

	for f in range(args.frames):
		t = f
		R = basis(vec(np.cos(np.pi / 4), 0, np.sin(np.pi / 4)), vec(0, 1, 0))
		cp = dp * t # camera pos
		theta = np.pi * f / 20
		cp = vec(np.sin(theta) * 2, 0, np.cos(theta))
		T = translate(cp)
		cam.set_frame(I[f])
		cam.set_transform(T)
		for v in verts:
			v_prime = xform(v, T)
			# print(f'actual point: {v}')
			cam.project(v, np.array([255, 255, 255], dtype=np.uint8))
			
		est_verts = pr.append(I[f], T)

		### Debug drawing
		cv2.putText(I[f], f"{f}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

		for feature in pr.X[-1].features:
			centroid, a, b = feature.covariance_ellipse()
			cv2.ellipse(I[f], (int(centroid[0]), int(centroid[1])), (int(a), int(b)), 0, 0, 360, (0, 255, 0), 1)

		# print(f'est_verts: {est_verts}')
		for v in est_verts:
			if v is None:
				continue
			v_prime = xform(v, T)
			cam.project(v, np.array([255, 0, 0], dtype=np.uint8))			

	imgs = [Image.fromarray(img).resize((256, 256), resample=Image.NEAREST) for img in I]
	imgs[0].save(args.output, save_all=True, append_images=imgs[1:], duration=10, loop=0)

	import os
	os.system("imgcat motion1.gif")

	