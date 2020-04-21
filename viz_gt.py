import os
import sys
import cv2
# import numpy as np


def vizGT(img_paths, gt_file):

	for i, img_path in enumerate(img_paths):
		img = cv2.imread(img_path)
		img_gt = gt_file[i]

		if img is not None and img_gt is not None:
			x, y, w, h = map(int, img_gt.split(","))
			cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
			cv2.imshow("bb frame", img)
			cv2.waitKey(100)


def main():
	img_dir = sys.argv[1]
	img_paths = [os.path.join(img_dir, img) for img in sorted(os.listdir(img_dir))]

	gt_file_path = sys.argv[2]
	gt_file = open(gt_file_path, 'r').readlines()

	vizGT(img_paths, gt_file)


if __name__ == '__main__':
	main()
