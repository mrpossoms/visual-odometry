import cv2
import numpy as np

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
			
			if p_sen_t == p_sen_t_1:
				# print(f"no displacement, can't compute 3D point {p_sen_t}, {p_sen_t_1}")
				continue
			
			_, y_basis_t_1 = self.camera.sensor_bases_in_world(transform=self.poses[-2])

			r_o_t_1, r_d_t_1 = self.camera.ray_in_world(p_sen_t_1[0], p_sen_t_1[1], transform=self.poses[-2])
			r_o_t, r_d_t = self.camera.ray_in_world(p_sen_t[0], p_sen_t[1], transform=self.poses[-1])

			# construct a plane which is parallel to r_d_t_1
			n = np.cross(r_d_t_1, y_basis_t_1)
			assert(np.dot(n, r_d_t_1) == 0)
			assert(np.dot(n, y_basis_t_1) == 0)
			o = r_o_t_1
			p = ray_plane_intersection(r_o_t, r_d_t, o, n)
			frame_point_estimates.append(p)
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

	