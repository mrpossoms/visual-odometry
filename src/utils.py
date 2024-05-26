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
	forward /= np.linalg.norm(forward)
	up /= np.linalg.norm(up)
	right = np.cross(forward, up)
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

def triangle(I, D, screen_points, tri_indices, verts, data, shader=None):
	m = [
		(screen_points[0] + screen_points[1]) / 2,
		(screen_points[1] + screen_points[2]) / 2,
		(screen_points[2] + screen_points[0]) / 2,
	]
	e = [
		screen_points[1] - screen_points[0],
		screen_points[2] - screen_points[1],
		screen_points[0] - screen_points[2],
	]
	corner_max = screen_points.max(axis=0)
	corner_min = screen_points.min(axis=0)
	n = [
		vec(e[0][1], -e[0][0]),
		vec(e[1][1], -e[1][0]),
		vec(e[2][1], -e[2][0]),
	]

	x1, y1 = screen_points[0]
	x2, y2 = screen_points[1]
	x3, y3 = screen_points[2]

	B = np.array([
		[x2*y3 - x3*y2, y2 - y3, x3 - x2],
		[x3*y1 - x1*y3, y3 - y1, x1 - x3],
		[x1*y2 - x2*y1, y1 - y2, x2 - x1],
	]) / (x1 * (y2-y3) + x2 * (y3-y1) + x3 * (y1-y2))

	for y in range(max(0, corner_min[1]), min(corner_max[1], I.shape[0]-1)):
		for x in range(max(0, corner_min[0]), min(corner_max[0], I.shape[1]-1)):
			p = vec(x,y)
			if np.dot(n[0], p -m[0]) >= 0 and np.dot(n[1], p - m[1]) >= 0 and np.dot(n[2], p - m[2]) >= 0:
				u, v, w = B @ vec(1, x, y)

				z = u * verts[0][2] + v * verts[1][2] + w * verts[2][2]
				if z < D[y,x]:
					D[y, x] = z
					if shader is not None:
						I[y, x] = shader(screen_points, tri_indices, u, v, w, data)
					else:
						I[y, x] = colors[0] * u + colors[1] * v + colors[2] * w

def tri_mesh(I, D, camera=None, verts=None, indices=None, data=None, shader=None, model=np.eye(4)):
	# assert(len(verts) == len(colors))
	assert(verts is not None)

	if indices is not None:
		assert(len(indices) % 3 == 0)
		for i in range(0, len(indices), 3):
			tri_inds = indices[i:i+3]
			points = np.array([camera.project_onto_sensor(xform(v, model)) for v in verts[tri_inds]])
			if None in points:
				return
			triangle(I, D, points, tri_inds, verts[tri_inds], data, shader=shader)
	else:
		assert(len(verts) % 3 == 0)
		for vi in range(0, len(verts), 3):
			points = np.array([camera.project_onto_sensor(v) for v in verts[vi:vi+3]])
			triangle(I, D, points, [vi,vi+1,vi+2], verts[vi:vi+3], data, shader=shader)