import time
import torch
import torch.utils.data as data
from torchvision import transforms

from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

import numpy as np
import numpy.random as random
from PIL import Image
from pathlib import Path

from data.ava_utils import load_boxes_and_labels, sec_to_frame
from data.ava_utils import gen_datalist, frame_to_sec
import math
import copy

"""
dataset for Atom Action Detection
"""
class data_repo(data.Dataset):

	def __init__(self,data_configs, is_train='train', 
				detect_thresh=0.0, use_subset=True, aug_transform=None):

		self.data_configs = data_configs
		self.is_train = is_train
		self.aug_transform = aug_transform

		self.valid_frames = range(self.data_configs["VALID_SEC_START"], self.data_configs["VALID_SEC_STOP"])

		if self.is_train == "train":
			csvfile = Path(self.data_configs["DATA_REPO"]) / self.data_configs["TRAIN_PREDICTED_CSV"]
			csvfile_gt = Path(self.data_configs["DATA_REPO"]) / self.data_configs["TRAIN_CSV"]
		elif self.is_train == "val":
			csvfile = Path(self.data_configs["DATA_REPO"]) / self.data_configs["VAL_PREDICTED_CSV"]
		else:
			raise NotImplementedError('Makelist not implemented')

		
		self._boxes_and_labels = load_boxes_and_labels(csvfile, 
												self.valid_frames, is_train,
												detect_thresh, use_subset )
		
		# Only need to train/eval on frames with boxes
		"""
		self.datalist is a list of frame-kits for dataloader to iterate,
		A single frame-kit Y in datalist has the following format:

		------------------------------------------------------------
		| video-id | mid-frame | wrapping-clip | boxes | boxes-labels |
		------------------------------------------------------------
		|   Y[0]   |   Y[1]	|	 Y[2]	  | Y[3] |	Y[4]	 |
		------------------------------------------------------------
		Y[0] = "0OLtK6SeTw"
		Y[1] = 27750
		Y[2] = [ 27743, 27744, ..., 27750, ..., 27758 ]
		Y[3] = [[0.5, 0.2, 0.8, 0.3], 
				[0.4, 0.3, 0.5 , 0.5], 
				... ]

		Y[4] = [[1], 
				[2, 3], 
				... ]
		"""
		self._datalist = gen_datalist(self._boxes_and_labels, self.data_configs, 
						self.valid_frames)


		if self.is_train == "train":
			self._boxes_and_labels_gt = load_boxes_and_labels(csvfile_gt, 
												self.valid_frames, is_train,
												detect_thresh, use_subset )
			self._datalist_gt = gen_datalist(self._boxes_and_labels_gt, 
										self.data_configs, 
										self.valid_frames)
			# Use both GT-box and Pred-Box in Training
			#self._datalist = self._datalist + self._datalist_gt
			self._datalist = self._datalist_gt

		# Conver self._datalist to single box-image list
		# Each element contain a single person
		self._datalist = self.gen_box_kit_list(self._datalist)


	def gen_box_kit_list(self, datalist):
		new_list = []
		for framekit in self._datalist:
			video_name = framekit[0]
			frame_idx  = framekit[1]
			clip	   = framekit[2]
			boxes	   = framekit[3]
			labels	   = framekit[4]
			for j, sig_box in enumerate(boxes):
				sig_label = labels[j]
				new_sigkit = [
					video_name,
					frame_idx,
					clip,
					[sig_box],
					[sig_label],
				]
				new_list.append(new_sigkit)
		return new_list

	def __len__(self):
		return len(self._datalist)

	def fetch_key(self, iters, batch_size):
		index_min = iters * batch_size
		index_max = index_min + batch_size
		index_max = min(index_max, len(self._datalist))
		boxes_batch = []
		vid_sec_ids = []
		for i in range(index_min, index_max):
			framekit = self._datalist[i]
			video_name = framekit[0]
			frame_idx  = framekit[1]
			boxes	   = framekit[3]
			sec = frame_to_sec(frame_idx, self.data_configs["TIME_OFFSET"], self.data_configs["FPS"])
			video_sec = "{}:{}".format(video_name, sec)
			vid_sec_ids.append(video_sec)
			boxes_batch.append(boxes)

		return boxes_batch, vid_sec_ids


	def __getitem__(self, index):

		if "RGB":
			#start_time = time.time()
			if self.is_train == "train" or self.is_train == "val":
				self.frames_repo = Path(self.data_configs["DATA_REPO"]) / self.data_configs["FRAMES_TRAIN"]
			else:
				self.frames_repo = Path(self.data_configs["DATA_REPO"]) / self.data_configs["FRAMES_TEST"]
			
			samples = self.fetch_rgb_samples(index)
			#elasped_time = time.time() - start_time
			#print("single-kit data-loading take {}s".format(elasped_time))
			return samples

		else:
			raise NotImplementedError('Optflow is not supported')
			return None


	def fetch_rgb_samples(self, index):
		framekit = self._datalist[index]

		video_name = framekit[0]
		frame_idx  = framekit[1]
		clip	   = framekit[2]
		boxes	   = framekit[3]
		labels	 = framekit[4]
		sec = frame_to_sec(frame_idx, self.data_configs["TIME_OFFSET"], self.data_configs["FPS"])
		video_sec = "{}:{}".format(video_name, sec)
		# Select Random the short edge to 256 ~ 320 pixels
		#max_size  = self.data_configs["FRAME_SHORT_MAX"]
		#min_size  = self.data_configs["FRAME_SHORT_MIN"]
		#test_size = self.data_configs["TEST_SHORT_SIZE"]
		#short_size = int(round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size)))

		# Read-Midframe
		midframe_file = Path(self.frames_repo) / video_name.strip() / '{}_{:06d}.jpg'.format(video_name, frame_idx) 

		# Read-Clip
		clip_pths = []
		### Temporal Jittering
		if self.data_configs["TRAIN_AUG_TIME_JITTER"]:
			jitter_range = min(self.data_configs["CLIP_M_FRAMES_AHEAD"], self.data_configs["CLIP_N_FRAMES_AFTER"]) // 2
			#jitter_rand = random.randint(jitter_range) * self.data_configs["CLIP_STEP"]
			jitter_rand = random.randint(jitter_range)
			if random.randint(2):
				jitter_rand = -jitter_rand
		else:
			jitter_rand = 0

		if self.is_train == "val":
			jitter_rand = 0

		start_frame = sec_to_frame(self.data_configs["VALID_SEC_START"],
								   self.data_configs["TIME_OFFSET"],
								   self.data_configs["FPS"]) + 1
		stop_frame = sec_to_frame(self.data_configs["VALID_SEC_STOP"],
								  self.data_configs["TIME_OFFSET"],
								  self.data_configs["FPS"])		

		for x in clip:
			frame_idx = min( max(start_frame, x + jitter_rand), stop_frame)
			frame_file = Path(self.frames_repo) / video_name.strip() / '{}_{:06d}.jpg'.format(video_name, frame_idx)
			clip_pths.append(frame_file)

		pil_clip_imgs 	 = read_clip(clip_pths)
		pil_midframe_img = read_image(midframe_file)

		#print(pil_midframe_img)
		# Image/Clip Augmentation (Substract Mean, Divide Std, RandomCrop, Filp etc...)
		if self.aug_transform != None:
			midframe_tensor, clip_tensor, new_boxes, labels = self.aug_transform(pil_midframe_img, pil_clip_imgs, copy.deepcopy(boxes), labels)
		else:
			print("Preprocess not specified")

		return midframe_tensor, clip_tensor, boxes, labels, video_sec


"""Read Image/Clip with original resolution"""
def read_clip(clip_pths):
	pil_clip_imgs = []
	for x in clip_pths:
		x_img = read_image(x)
		pil_clip_imgs.append(x_img)
	return pil_clip_imgs


def read_image(img_pth):
	pil_img = Image.open(img_pth).convert('RGB')
	return pil_img


#"""Read Image with Resizing, but keep ratio"""
#def read_clip_short_side_resize(clip_pths, short_size):
#	pil_clip_imgs = []
#	for x in clip_pths:
#		x_img = read_image_short_side_resize(x, short_size)
#		pil_clip_imgs.append(x_img)
#
#	return pil_clip_imgs


#def read_clip_TEST_short_side(clip_pths, test_size=256):
#	# Select Random the short edge to 256 ~ 320 pixels
#	short_size = test_size
#
#	pil_clip_imgs = []
#	for x in clip_pths:
#		x_img = read_image_short_side_resize(x, short_size)
#		pil_clip_imgs.append(x_img)
#
#	return pil_clip_imgs


#def read_image_short_side_resize(img_pth, short_size):
#	pil_img = Image.open(img_pth).convert('RGB')
#	width, height = pil_img.size
#	if width > height:
#		new_height = short_size
#		new_width  = int(math.floor((float(width) / height) * new_height))
#	else:
#		new_width  = short_size
#		new_height = int(math.floor((float(height) / width) * new_width))
#	resize_img = pil_img.resize((new_width, new_height), Image.ANTIALIAS)
#
#	return resize_img




