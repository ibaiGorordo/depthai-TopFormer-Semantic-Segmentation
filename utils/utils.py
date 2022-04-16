from time import monotonic
import numpy as np
import cv2
import os

dir_path = os.path.dirname(os.path.abspath(__file__))
ade20k_info_path = dir_path + "/ade20k_label_colors.txt"

def read_ade20k_info(info_path=ade20k_info_path):
	with open(info_path) as fp:
		lines = fp.readlines()

		labels = [line[:-1].replace(';', ',').split(',')[0] for line in lines]
		colors = np.array([line[:-1].replace(';', ',').split(',')[-3:] for line in lines]).astype(int)

	return colors, labels

colors, labels = read_ade20k_info()
	
def util_draw_seg(seg_map, image, alpha = 0.5):

	# Convert segmentation prediction to colors
	color_segmap = cv2.resize(image, (seg_map.shape[1], seg_map.shape[0]), cv2.INTER_AREA)
	color_segmap[seg_map>0] = colors[seg_map[seg_map>0]]

	# Resize to match the image shape
	color_segmap = cv2.resize(color_segmap, (image.shape[1],image.shape[0]), cv2.INTER_CUBIC)

	# Fuse both images
	if(alpha == 0):
		combined_img = np.hstack((image, color_segmap))
	else:
		combined_img = cv2.addWeighted(image, alpha, color_segmap, (1-alpha),0)

	return combined_img


class FpsUpdater(object):
    """docstring for FpsUpdater"""
    def __init__(self):
        super(FpsUpdater, self).__init__()

        self.start_time = monotonic()
        self.counter = 0
        self.fps = 0

    def __call__(self):
        # Update fps
        self.counter+=1
        time_now = monotonic() 
        if (time_now - self.start_time) > 1 :
            self.fps = self.counter / (time_now - self.start_time)
            self.counter = 0
            self.start_time = time_now

        return self.fps
        