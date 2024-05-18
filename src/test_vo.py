from vo import *

def test_pinhole_geometry():
	I = np.zeros((11, 11), dtype=np.uint8)

	cam = Pinhole(I, focal_length_m=1, receptor_size_m=0.1)
	sensor_coord = cam.frame_coord_in_world(11//2, 11//2)
	assert((sensor_coord == vec(0, 0, -1)).all())


def test_pinhole_geometry_translation():
	I = np.zeros((11, 11), dtype=np.uint8)

	cam = Pinhole(I, focal_length_m=1, receptor_size_m=0.1)
	cam.set_transform(translate(vec(0, 0, 1)))
	sensor_coord = cam.frame_coord_in_world(11//2, 11//2)
	assert((sensor_coord == vec(0, 0, 0)).all())

def test_pinhole_sensor_bases():
	I = np.zeros((11, 11), dtype=np.uint8)

	cam = Pinhole(I, focal_length_m=1, receptor_size_m=0.1)
	cam.set_transform(translate(vec(0, 0, 1)))

	x, y = cam.sensor_bases_in_world()

	assert((x == vec(1, 0, 0)).all())
	assert((y == vec(0, 1, 0)).all())

def test_pinhole_sensor_bases_rotated():
	I = np.zeros((11, 11), dtype=np.uint8)

	cam = Pinhole(I, focal_length_m=1, receptor_size_m=0.1)
	cam.set_transform(basis(vec(1, 0, 0), vec(0, 1, 0)))

	x, y = cam.sensor_bases_in_world()

	assert((x == vec(0, 0, 1)).all())
	assert((y == vec(0, 1, 0)).all())

def test_pinhole_project_onto_sensor():
	I = np.zeros((11, 11), dtype=np.uint8)

	cam = Pinhole(I, focal_length_m=1, receptor_size_m=0.1)
	# cam.set_transform(basis(vec(1, 0, 0), vec(0, 1, 0)))

	p = cam.project_onto_sensor(vec(0, 0, -1))

	assert(p is None)

def test_pinhole_ray():
	I = np.zeros((11, 11), dtype=np.uint8)

	cam = Pinhole(I, focal_length_m=1, receptor_size_m=0.1)
	# cam.set_transform(basis(vec(1, 0, 0), vec(0, 1, 0)))

	x, y = cam.sensor_bases_in_world()
	o, d = cam.ray_in_world(0, 5)

	assert((o == x * (-cam.sensor_dims[0] / 2) + cam.sensor_origin_in_world()).all())

	assert(np.dot(d, vec(0, 0, 1)) > 0)

def test_ray_plane_intersection():

	# ray is parallel to the plane
	o = vec(0, 0, 0)
	d = vec(1, 0, 0)
	p = vec(0, 0, 1)
	n = vec(0, 0, 1)

	i = ray_plane_intersection(o, d, p, n)
	assert(i == None)

	# ray is perpendicular to the plane
	o = vec(0, 0, 0)
	d = vec(0, 0, 1)
	p = vec(0, 0, 2)
	n = vec(0, 0, -1)

	i = ray_plane_intersection(o, d, p, n)
	assert((i == vec(0, 0, 2)).all())

	# ray is off-axis to the plane, aiming at origin
	p = vec(0, 0, 2)
	n = vec(0, 0, -1)
	o = vec(1, 33, 7)
	d = p - o

	i = ray_plane_intersection(o, d, p, n)
	assert((i == vec(0, 0, 2)).all())

	# ray is off-axis to the plane
	p = vec(0, 0, 2)
	n = vec(0, 0, 2)
	o = vec(1, 33, 7)
	d = vec(0, 0, -1)

	i = ray_plane_intersection(o, d, p, n)
	assert((i == vec(1, 33, 2)).all())


	# ray is not perpendicular to the plane
	o = vec(0, 0, 0)
	d = vec(0, 0, -1)
	p = vec(0, 0, 1)
	n = vec(0, 0, -1)

	i = ray_plane_intersection(o, d, p, n)
	assert(i == None)