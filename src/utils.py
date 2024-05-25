import numpy as np
from .pinhole import Pinhole

def point_from_motion(camera, p_sen_t_1, p_sen_t, pose_t_1, pose_t):
	if (p_sen_t == p_sen_t_1).all():
		# print(f"no displacement, can't compute 3D point {p_sen_t}, {p_sen_t_1}")
		# return None
		r_o_t, r_d_t = camera.ray_in_world(p_sen_t[0], p_sen_t[1], transform=pose_t)
		return r_o_t + r_d_t * 100
	
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
	for z in np.arange(-1,1+d,d):
		for x in np.arange(-1,1+d,d):
			for y in np.arange(-1,1+d,d):
				cv.append(vec(x,y,z) + position)
				if color_for_coord is not None:
					cc.append(color_for_coord(vec(x,y,z)))
	return cv, cc

def line_samples(p1, p2): #  -> list[tuple(int,int,float)]:
	# draw a line from p1 to p2
	# with color interpolation
	# p1, p2 are 2d pixel coordinates
	# color1, color2 are 3d color values
	# we'll interpolate between them
	# and draw the line
	# Bresenham's line algorithm
	x0, y0 = p1
	x1, y1 = p2
	x, y = x0, y0
	dx = abs(x1 - x0)
	dy = abs(y1 - y0)
	sx = 1 if x0 < x1 else -1
	sy = 1 if y0 < y1 else -1
	err = dx - dy

	samples = []

	d = abs(x1 - x0) + abs(y1 - y0)
	while True:
		# draw the pixel
		# with color interpolation
		# between color1 and color2
		# at the current position
		t = abs(x1 - x) + abs(y1 - y)
		samples.append((x, y, t / d))

		if x == x1 and y == y1:
			break

		e2 = 2 * err
		if e2 > -dy:
			err -= dy
			x += sx
		if e2 < dx:
			err += dx
			y += sy

	return samples

def line(I, p1, p2, color1, color2):
	# draw a line from p1 to p2
	# with color interpolation
	# p1, p2 are 2d pixel coordinates
	# color1, color2 are 3d color values
	# we'll interpolate between them
	# and draw the line
	# Bresenham's line algorithm
	x0, y0 = p1
	x1, y1 = p2
	x, y = x0, y0
	dx = abs(x1 - x0)
	dy = abs(y1 - y0)
	sx = 1 if x0 < x1 else -1
	sy = 1 if y0 < y1 else -1
	err = dx - dy

	d = abs(x1 - x0) + abs(y1 - y0)
	while True:
		# draw the pixel
		# with color interpolation
		# between color1 and color2
		# at the current position
		t = abs(x1 - x) + abs(y1 - y)
		I[y, x] = color1 #(color1 * (t / d) + color2 * (1 - t / d))

		if x == x1 and y == y1:
			break

		e2 = 2 * err
		if e2 > -dy:
			err -= dy
			x += sx
		if e2 < dx:
			err += dx
			y += sy

def triangle(I, points, color):
	center = np.average(points, axis=0)
	m = [
		(points[0] + points[1]) / 2,
		(points[1] + points[2]) / 2,
		(points[2] + points[0]) / 2,
	]
	e = [
		points[1] - points[0],
		points[2] - points[1],
		points[0] - points[2],
	]
	corner_max = points.max(axis=0)
	corner_min = points.min(axis=0)

	n = [
		vec(e[0][1], -e[0][0]),
		vec(e[1][1], -e[1][0]),
		vec(e[2][1], -e[2][0]),
	]

	for y in range(corner_min[1], corner_max[1]):
		for x in range(corner_min[0], corner_max[0]):
			p = vec(x, y)
			if np.dot(n[0], p - m[0]) > 0 and np.dot(n[1], p - m[1]) > 0 and np.dot(n[2], p - m[2]) > 0:
				I[y, x] = color