import cv2
import numpy as np
from kf import KalmanFilter


class KFPR:
	def __init__(self, camera):
		self.camera = camera
		self.frames = []
		self.poses = []
		self.descriptors = []
		self.keypoints = []
		self.matches = []
		self.point_estimates = []

		self.orb = cv2.ORB_create(patchSize=5, nlevels=1)
		self.matcher = cv2.BFMatcher() #cv2.NORM_HAMMING, crossCheck=True)


	def reset(self):
		self.frames = []
		self.cam_pose = []
		self.descriptors = []
		self.keypoints = []
		self.matches = []
		self.poses = []

	def size(self):
		nf = len(self.frames)
		np = len(self.poses)
		nd = len(self.descriptors)
		nk = len(self.keypoints)
		nm = len(self.matches)
		assert(nf == np and nf == nd and nf == nk and nf == nm)
		return nf

	def append(self, frame, cam_pose, max_distance=None) -> list[np.array]:
		self.frames += [frame]
		self.poses += [cam_pose]
		keypoints, descriptors = self.orb.detectAndCompute(frame, None)

		if keypoints is None or descriptors is None:
			self.point_estimates += [[]]
			self.matches += [[]]
			return			

		self.keypoints += [keypoints]
		self.descriptors += [np.array(descriptors)]

		frame_point_estimates = []

		if len(self.frames) < 2 or len(self.descriptors[-1]) == 0 or len(self.descriptors[-2]) == 0:
			self.point_estimates += [frame_point_estimates]
			self.matches += [[]]
			return

		frame_matches = sorted(self.matcher.match(self.descriptors[-1], self.descriptors[-2]), key = lambda x:x.distance)
		if len(frame_matches) == 0:
			frame_matches = []
		self.matches += [frame_matches]

		for match in self.matches[-1]:
			if max_distance is not None and match.distance > max_distance:
				continue

			kps_t_1 = self.keypoints[-2]
			kps_t = self.keypoints[-1]

			# print(f'kpt_t_1: {len(kps_t_1)}, kpt_t: {len(kps_t)}')
			# print(f"match indices: {match.queryIdx}, {match.trainIdx}")

			assert(match.queryIdx < len(kps_t))
			assert(match.trainIdx < len(kps_t_1))

			p_sen_t_1 = kps_t_1[match.trainIdx].pt			
			p_sen_t = kps_t[match.queryIdx].pt
			
			p = point_from_motion(self.camera, p_sen_t_1, p_sen_t, self.poses[-2], self.poses[-1])

			if p is None:
				continue

			frame_point_estimates.append(p)


		self.point_estimates += [frame_point_estimates]

		return frame_point_estimates


if __name__ == "__main__":
	import argparse
	from PIL import Image
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument("--output", type=str, default="motion.gif")
	arg_parser.add_argument("--frames", type=int, default=25)
	args = arg_parser.parse_args()
	
	I = np.zeros((args.frames, 128, 128, 3), dtype=np.uint8)

	# verts = [vec(1,1,1) + vec(0, 0, 5), vec(-1,2,1) + vec(0, 0, 10)]
	verts = cube(d=2, position=vec(0, 0, 10))

	dp = vec(0, 0, 0.5)
	cam = Pinhole(I[0])
	vo = VO(cam)

	for f in range(args.frames):
		t = f
		R = basis(vec(np.cos(np.pi / 4), 0, np.sin(np.pi / 4)), vec(0, 1, 0))
		cp = dp * t # camera pos
		T = translate(cp)
		cam.set_frame(I[f])
		cam.set_transform(T)
		for v in verts:
			v_prime = xform(v, T)
			# print(f'actual point: {v}')
			cam.project(v, np.array([0, 255, 0], dtype=np.uint8))
			
		est_verts = vo.append(I[f], T)
		print(f'est_verts: {est_verts}')
		if est_verts is not None:
			for v in est_verts:
				if v is None:
					continue
				v_prime = xform(v, T)
				cam.project(v, np.array([255, 0, 0], dtype=np.uint8))

	imgs = [Image.fromarray(img).resize((256, 256), resample=Image.NEAREST) for img in I]
	imgs[0].save(args.output, save_all=True, append_images=imgs[1:], duration=300, loop=0)

	import os


	