import cv2
import numpy as np
from utils import *

class VO:
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
		# import pdb; pdb.set_trace()
		keypoints, descriptors = self.orb.detectAndCompute(frame, None)

		if keypoints == () or len(descriptors) == 0:
			self.point_estimates += [[]]
			self.keypoints += [[]]
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
	import os
	import argparse
	from PIL import Image
	from pinhole import Pinhole
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument("--output", type=str, default="motion.gif")
	arg_parser.add_argument("--frames", type=int, default=25)
	args = arg_parser.parse_args()
		
	I = np.zeros((args.frames, 128, 128, 3), dtype=np.uint8) * 255

	def cube_colors(v):
		return (v * 32 + (128 + 32)).astype(np.uint8)

	verts, colors = cube(d=2, position=vec(0, 0, 10), color_for_coord=cube_colors)
	verts, colors = verts[1:2], colors[1:2]

	dp = vec(0, 0, 0.5)
	cam = Pinhole(I[0])
	vo = VO(cam)

	assert(vo.size() == 0)

	noise = (np.random.default_rng().random((32,3,3,3))) # * 32).astype(np.uint8)

	for f in range(args.frames):
		cp = dp * (f+4) # camera pos
		T = translate(cp)
		cam.set_frame(I[f])
		cam.set_transform(T)

		# render verts to camera frame
		for i, (v, c) in enumerate(zip(verts, colors)):
			coord = cam.project(v, c)
			if coord is not None:
				# draw a little patch with noise around the point
				I[f,coord[1]-1:coord[1]+2,coord[0]-1:coord[0]+2] = (c * noise[i]).astype(np.uint8)       

		# hand the frame and transform to VO instance, produces estimates
		est_verts = vo.append(I[f], T)

		# render location of reconstructed verts into frame
		if est_verts is not None:
			for v in est_verts:
				if v is None:
					continue
				coord = cam.project(v, None)
				cv2.circle(I[f],(coord[0],coord[1]),5,(255, 128, 0),1)

		# show_features(I[f], vo.keypoints[f])
		show_deltas(vo, index=f)
		
		cv2.putText(I[f], f"f:{f} kps:{len(vo.keypoints[f])}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
		
	imgs = [Image.fromarray(
		img).resize((256, 256), resample=Image.NEAREST) for img in I]
	imgs[0].save("motion.gif", save_all=True, append_images=imgs[1:], duration=10, loop=0)

	os.system("imgcat motion.gif")

	