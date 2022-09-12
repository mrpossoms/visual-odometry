import cv2


class VO:

	frames = []
	descriptors = []
	keypoints = []

	orb = cv2.ORB_create()
	matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	def __init__(self):
		pass

	def append(self, frame):
		self.frames += [frame]

		keypoints = self.orb.detect(frame, None)
		keypoints, descriptors = self.orb.compute(frame, keypoints)

		self.keypoints += [keypoints]
		self.descriptors += [descriptors]

		if len(self.frames) < 2:
			return

		matches = self.matcher.match(self.descriptors[-2], self.descriptors[-1])

		# roll off old frames
		self.frames = self.frames[1:]



