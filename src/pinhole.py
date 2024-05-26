import numpy as np


class Pinhole:
	def __init__(self, sensor: np.array, receptor_size_m=0.0001, focal_length_m=0.005):
		self._f = focal_length_m
		self._cam_transform = np.eye(4)
		self._world_transform = np.eye(4)
		self._sensor = sensor
		self.sensor_dims = np.array(sensor.shape[:2]) * receptor_size_m
		assert(self.sensor_dims.size == 2)
		self._receptor_dims = np.array([receptor_size_m, receptor_size_m])
		self._sensor_extents = [
			-self.sensor_dims / 2,
			self.sensor_dims / 2,
		]

	def projection(self, z=None) -> np.array:
		if z is None:
			return np.array([
				[1,   0,         0, 0],
				[0,   1,         0, 0],
				[0,   0,         1, 0],
				[0,   0,-1/self._f, 0],
			])
		else:
			w = -z/self._f
			return np.array([
				[1/w,   0, 0, 0],
				[  0, 1/w, 0, 0],
			])

	def project_onto_sensor(self, p: np.array) -> np.array:
		# transform point into view space
		p_prime = np.matmul(self._cam_transform, np.append(p, [1]))

		if p_prime[2] <= 0:
			return None

		P = self.projection(z=p_prime[2])
		sensor_plane_pos = None

		if P.shape[0] == 2:
			sensor_plane_pos = np.matmul(P, p_prime)
		else:
			p_prime = np.matmul(P, p_prime)		
			if p_prime[2] <= 0:
				return None
			p_prime /= p_prime[3]
			sensor_plane_pos = p_prime[:2]

		# dimensionality constraint checks
		assert(sensor_plane_pos.size == 2)
		assert(self._sensor_extents[0].size == 2)

		sensor_coord = ((sensor_plane_pos - self._sensor_extents[0]) / self.sensor_dims) * np.array(self._sensor.shape[:2])

		return sensor_coord.astype(int)

	def project(self, p: np.array, c: np.array):
		sensor_coord = self.project_onto_sensor(p)

		# bounds check
		if ((sensor_coord[0] < 0 or sensor_coord[1] < 0) or
		    (sensor_coord[0] >= self._sensor.shape[0] or sensor_coord[1] >= self._sensor.shape[1])):
			return None

		if sensor_coord is not None and c is not None:
			self._sensor[sensor_coord[1], sensor_coord[0]] = c

		return sensor_coord

	def draw_mesh(self, verts, colors, tris):

		for tri in tris:
			tri_verts = verts[tri]
			tri_colors = colors[tri]
			pts = [self.project(v, c).astype(np.int) for v, c in zip(tri_verts, tri_colors)]

			#draw the triangle as a series of lines


	def set_transform(self, transform: np.array):
		self._world_transform = transform
		self._cam_transform = np.linalg.inv(transform)

	def set_frame(self, frame: np.array):
		self._sensor = frame

	def aperture_in_world(self, transform=None) -> np.array:
		if transform is None:
			transform = self._world_transform
		return np.matmul(transform, np.array([0, 0, 0, 1]))[:3]

	def sensor_origin_in_world(self) -> np.array:
		return np.matmul(self._world_transform, np.array([0, 0, -self._f, 1]))[:3]

	def sensor_bases_in_world(self, transform=None) -> tuple[np.array, np.array]:
		# the sensor is a rectangle in the xy plane
		# with the normal pointing in the -z direction
		# so the basis vectors are just the first two columns
		# of the world transform
		if transform is None:
			transform = self._world_transform

		return (
			np.matmul(transform, np.array([1, 0, 0, 0]))[:3],
			np.matmul(transform, np.array([0, 1, 0, 0]))[:3],
		)

	def frame_coord_in_world(self, x: int, y: int, transform=None) -> np.array:
		# x, y are the pixel coordinates of the sensor
		# we assume the sensor is at (0, 0, -f)
		# and the camera is looking down the z-axis
		# so the ray is just the x, y coordinates of the sensor
		# scaled by the focal length
		if transform is None:
			transform = self._world_transform
		sensor_point = np.array([
			(x / (self._sensor.shape[0] - 1)) * self.sensor_dims[0] + self._sensor_extents[0][0],
			(y / (self._sensor.shape[1] - 1)) * self.sensor_dims[1] + self._sensor_extents[0][1],
			-self._f,
		]) 
		return np.matmul(transform, np.append(sensor_point, [1]))[:3]

	def ray(self, x: int, y: int) -> np.array:
		# x, y are the pixel coordinates of the sensor
		# we assume the sensor is at (0, 0, -f)
		# and the camera is looking down the z-axis
		# so the ray is just the x, y coordinates of the sensor
		# scaled by the focal length
		sensor_point = np.array([
			(x / (self._sensor.shape[0] - 1)) * self.sensor_dims[0] + self._sensor_extents[0][0],
			(y / (self._sensor.shape[1] - 1)) * self.sensor_dims[1] + self._sensor_extents[0][1],
			-self._f,
		]) + self._receptor_dims / 2
		return -sensor_point / np.linalg.norm(sensor_point)

	def ray_in_world(self, x: int, y: int, transform=None) -> tuple[np.array, np.array]:
		o = self.frame_coord_in_world(x, y, transform=transform)
		d = self.aperture_in_world(transform=transform) - o
		return (o, d)