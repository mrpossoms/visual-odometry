import cv2
import numpy as np
from kf import KalmanFilter

def point_from_motion(camera, p_sen_t_1, p_sen_t, pose_t_1, pose_t):
	if p_sen_t == p_sen_t_1:
		# print(f"no displacement, can't compute 3D point {p_sen_t}, {p_sen_t_1}")
		return None
	
	_, y_basis_t_1 = camera.sensor_bases_in_world(transform=pose_t_1)

	r_o_t_1, r_d_t_1 = camera.ray_in_world(p_sen_t_1[0], p_sen_t_1[1], transform=pose_t_1)
	r_o_t, r_d_t = camera.ray_in_world(p_sen_t[0], p_sen_t[1], transform=pose_t)

	# construct a plane which is parallel to r_d_t_1
	n = np.cross(r_d_t_1, y_basis_t_1)
	assert(np.dot(n, r_d_t_1) == 0)
	assert(np.dot(n, y_basis_t_1) == 0)
	o = r_o_t_1
	return ray_plane_intersection(r_o_t, r_d_t, o, n)

class Feature:
	def __init__(self, camera: Pinhole, keypoints: list[cv2.KeyPoint], poses: list[np.array]):
		assert(len(keypoints) >= 2)
		assert(len(poses) >= 2)

		self._cam = camera
		self._keypoints = keypoints
		self._poses = poses

		x0 = point_from_motion(camera, keypoints[-2].pt, keypoints[-1].pt, poses[-2], poses[-1])
		assert(x0)

		self._kf = KalmanFilter(
			F=np.eye(4),
			B=poses[-1] @ np.linalg.inv(poses[-2]),
			H=camera.projection(z=1),
			Q=np.eye(4), # proc noise TDB
			R=np.eye(2), # meas noise TDB
			P=np.eye(4), # state est covar TDB
			x=x0
			)

	def filter_keypoints(self, keypoints: list[cv2.KeyPoint]) -> list[cv2.KeyPoint]:
		# Use the state estimate covariance to project an ellipse into the sensor's xy plane
		# use that ellipse to determine if a keypoint should be considered part of this feature
		centroid = self._cam.project(self._kf.x)
		extents = self._cam.project(self._kf.x[:3] + self._kf.P.diagonal[:3]) - centroid


		for kp in keypoints:
			

	def update(self, keypoints: list[cv2.KeyPoint], pose: np.array):




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
			# if p_sen_t == p_sen_t_1:
			# 	# print(f"no displacement, can't compute 3D point {p_sen_t}, {p_sen_t_1}")
			# 	continue
			
			# _, y_basis_t_1 = self.camera.sensor_bases_in_world(transform=self.poses[-2])

			# r_o_t_1, r_d_t_1 = self.camera.ray_in_world(p_sen_t_1[0], p_sen_t_1[1], transform=self.poses[-2])
			# r_o_t, r_d_t = self.camera.ray_in_world(p_sen_t[0], p_sen_t[1], transform=self.poses[-1])

			# # construct a plane which is parallel to r_d_t_1
			# n = np.cross(r_d_t_1, y_basis_t_1)
			# assert(np.dot(n, r_d_t_1) == 0)
			# assert(np.dot(n, y_basis_t_1) == 0)
			# o = r_o_t_1
			# p = ray_plane_intersection(r_o_t, r_d_t, o, n)
			# frame_point_estimates.append(p)
			# print(f'p: {p}')

		self.point_estimates += [frame_point_estimates]

		return frame_point_estimates



def show_features(frame, features):
	frame_rgb = frame
	if len(frame_rgb.shape) == 2:
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
	
	for i in features:
		x,y = i.pt
		cv2.circle(frame_rgb,(int(x),int(y)),5,255,1)
	
	return frame_rgb

def show_deltas(vo, max_distance=None, index=-1):
	frame_rgb = vo.frames[index]
	if len(frame_rgb.shape) == 2:
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
	
	for match in vo.matches[index][:8]:
		if max_distance is not None and match.distance > max_distance:
			continue

		kp_t_1 = vo.keypoints[index-1]
		kp_t = vo.keypoints[index]
		p1 = kp_t[match.queryIdx].pt
		p2 = kp_t_1[match.trainIdx].pt
		cv2.line(frame_rgb, 
				 (int(p1[0]), int(p1[1])), 
				 (int(p2[0]), int(p2[1])),
				 (0,255,0),1)
		
	return frame_rgb

def draw_cross(frame, coord, color=(255, 128, 0)):
	if coord is None:
		return

	for i in range(-3, 3):
		frame[i+coord[0],coord[1]] = color
		frame[coord[0],i+coord[1]] = color


def ray_plane_intersection(r_o, r_d, p_o, p_n):
	# r_o is the origin of the ray
	# r_d is the direction of the ray
	# p_o is a point on the plane
	# p_n is the normal of the plane
	# returns the point of intersection
	den = np.dot(r_d, p_n)
	if den == 0:
		return None
	
	t = np.dot(p_o - r_o, p_n) / den

	if t < 0:
		return None

	return r_o + t * r_d

def vec(*argv):
	return np.array([*argv])

def xform(v, M):
	v_aug = np.append(v, [1])
	return np.matmul(M, v_aug)[:3]

def basis(forward, up):
	right = np.cross(forward / np.linalg.norm(forward), up / np.linalg.norm(up))
	up = np.cross(forward, -right)
	return np.array([
		np.append(right, 0),
		np.append(up, 0),
		np.append(forward, 0),
		[0, 0, 0, 1]
	]).T

def translate(t):
	return np.array([
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 1, 0],
		np.append(t, 1),
	]).T

def cube(d=1, position=vec(0,0,0), color_for_coord=None):
	cv, cc = [], []
	for x in np.arange(-1,1+d,d):
		for y in np.arange(-1,1+d,d):
			for z in np.arange(-1,1+d,d):
				cv.append(vec(x,y,z) + position)
				if color_for_coord is not None:
					cc.append(color_for_coord(vec(x,y,z)))
	return cv, cc


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


	