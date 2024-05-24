import numpy as np
from .pinhole import Pinhole

def point_from_motion(camera, p_sen_t_1, p_sen_t, pose_t_1, pose_t):
	if (p_sen_t == p_sen_t_1).all():
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