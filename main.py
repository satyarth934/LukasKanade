from __future__ import print_function
import os
import re
import sys
import cv2
import copy
# import numpy as np
# from scipy.interpolate import RectBivariateSpline
sys.dont_write_bytecode = True

import lk
# import utils


viz = True
make_video = False


def lkPipeline(path, gt_path):
	gt_rects = [map(int, re.split(';|,| |\t', (line.strip())))
	                for line in open(gt_path).readlines()]
	gt_rects = [[x, y, x + w, y + h] for x, y, w, h in gt_rects]

	cv_img = []
	img_paths = [os.path.join(path, img) for img in sorted(os.listdir(path))]
	cv_img = [cv2.imread(img_path) for img_path in img_paths]

	# rectangle = [118, 100, 338, 280]
	# rectangle = [269, 75, (269 + 34), (75 + 64)] 	# bolt
	# rectangle = [70, 51, (70 + 107), (51 + 87)] 	# car
	rectangle = gt_rects[0]
	b = rectangle[3] - rectangle[1]
	l = rectangle[2] - rectangle[0]
	rectangle0 = copy.deepcopy(rectangle)
	capture_in = cv_img[0]
	capture_gray_in = cv2.cvtColor(capture_in, cv2.COLOR_BGR2GRAY)
	capture_gray_in = cv2.equalizeHist(capture_gray_in)
	if make_video:
		out = cv2.VideoWriter('Car1112.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, capture_in.shape[:2][::-1])

	for i in range(0, len(cv_img) - 1):
		print('%d of %d' % (i, len(cv_img)))

		# Updating the template bounding box every 50 frames.
		# This ensures that we do not lose the object of interest.
		# In a practical environment, this is the output of detection module.
		if i % 50 == 0:

			rectangle = gt_rects[i]

			l = rectangle[2] - rectangle[0]
			b = rectangle[3] - rectangle[1]

			rectangle0 = copy.deepcopy(rectangle)
			capture_in = cv_img[i]
			capture_gray_in = cv2.cvtColor(capture_in, cv2.COLOR_BGR2GRAY)
			capture_gray_in = cv2.equalizeHist(capture_gray_in)

		capture = cv_img[i]
		maincapture = capture
		capture_gray = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
		capture_gray = cv2.equalizeHist(capture_gray)
		cv2.rectangle(maincapture, (int(rectangle[0]), int(rectangle[1])), (int(
			rectangle[0]) + l, int(rectangle[1]) + b), (255, 0, 0), 3)
		if viz:
			cv2.imshow('Tracking Car', capture)
		if make_video:
			out.write(capture)

		capture_next = cv_img[i + 1]
		capture_gray_next = cv2.cvtColor(capture_next, cv2.COLOR_BGR2GRAY)
		capture_gray_next = cv2.equalizeHist(capture_gray_next)

		in_temp_x = capture_gray_in / 255.
		in_temp = capture_gray / 255.
		in_temp_a = capture_gray_next / 255.
		stop = lk.LucasKanade(in_temp_x, in_temp_a, rectangle0)
		rectangle[0] = stop[0] + rectangle0[0]
		rectangle[1] = stop[1] + rectangle0[1]
		rectangle[2] = stop[0] + rectangle0[2]
		rectangle[3] = stop[1] + rectangle0[3]

		if viz:
			if cv2.waitKey(1) == 27:
				cv2.destroyAllWindows()
				break


def main():
	path = '../Data/Car4/img/'
	# path = '../Data/Bolt2/img/'
	# path = '../Data/DragonBaby/DragonBaby/img/'
	gt_path = '../Data/Car4/groundtruth_rect.txt'
	# gt_path = '../Data/Bolt2/groundtruth_rect.txt'
	# gt_path = '../Data/DragonBaby/DragonBaby/groundtruth_rect.txt'

	lkPipeline(path, gt_path)


if __name__ == '__main__':
	main()
