'''
Dataset for Kinetics, Basketball-8, ...
'''
from torchvision import transforms 
import torch
from pathlib import Path
from PIL import Image
import random
import numpy as np
import torch.utils.data as data

class DataRepo(data.Dataset):
	def __init__(self, cfg, is_train='train', aug=None):
		self.is_train=is_train
		self.use_subset   = cfg['USE_SUBSET']
		self.frame_repo   = cfg['FRAME_REPO']
		self.frame_format = cfg['FRAME_FORMAT']
		self.t_size       = cfg['T_SIZE']
		self.t_step       = cfg['T_STEP']
		self.dense_or_uniform = cfg['DENSE_OR_UNIFORM']
		self.uniform_twice = cfg['UNIFORM_TWICE']
		self.val_test_aug =  cfg["VAL_TEST_AUG"]
		self.val_flip = cfg["VAL_FLIP"]		
		self.sample_times = cfg["VAL_SAMPLE_TIMES"]

		if self.is_train == "train":
			self.list_txt = cfg['TRN_LIST_TXT']
		elif self.is_train == "val":
			self.list_txt = cfg['VAL_LIST_TXT']
		else:
			self.list_txt = cfg['TEST_LIST_TXT']
		self.aug = aug
		self.vidpth_framecnt_label = load_annot_txt(self.list_txt,
													self.use_subset)


	def __len__(self):
		return len(self.vidpth_framecnt_label)

	
	def __getitem__(self, index):
		samples = self.fetch_rgb_clip(index)
		return samples	

	def fetch_rgb_clip(self, index):
		vidpth   = self.vidpth_framecnt_label[index][0]		
		framecnt = self.vidpth_framecnt_label[index][1]
		label    = self.vidpth_framecnt_label[index][2]

		frame_vid_folder = Path(self.frame_repo) / vidpth

		# Train Data Process
		if self.is_train == "train":
			if self.dense_or_uniform == "uniform":
				pil_clip  = read_random_uniform_clip(frame_vid_folder, framecnt,
									self.frame_format,
									self.t_size)
			else:
				pil_clip  = read_random_clip(frame_vid_folder, framecnt,
									self.frame_format, self.t_size, self.t_step)

			assert self.aug != None
			tensor_clip, label = self.aug(pil_clip, label)

		# Test/Val Data Process
		else:
			assert self.aug !=None
			
			if self.dense_or_uniform == "uniform":
				tmp_repo = []
				pil_clip_repo  = read_uniform_clip(frame_vid_folder, framecnt,
											self.frame_format,
											self.t_size, self.uniform_twice)
				tmp_repo = []
				for pil_clip in pil_clip_repo:
					# Original
					tensor_clip, label = self.aug(pil_clip, label)
					tmp_repo.append(tensor_clip)
					# Flip
					if self.val_flip:
						pil_clip = [ transforms.functional.hflip(x) for x in pil_clip ]
						tensor_clip, label = self.aug(pil_clip, label)
						tmp_repo.append(tensor_clip)
						
				tensor_clip = torch.cat(tmp_repo, 0)

			else:
				pil_clip_repo = read_clip(frame_vid_folder, framecnt,
										self.frame_format, self.t_size, self.t_step,
										self.sample_times)
				tmp_repo = []
				for pil_clip in pil_clip_repo:
					# Original
					tensor_clip, label = self.aug(pil_clip, label)
					tmp_repo.append(tensor_clip)
					# Flip
					if self.val_flip:
						pil_clip = [ transforms.functional.hflip(x) for x in pil_clip ]
						tensor_clip, label = self.aug(pil_clip, label)
						tmp_repo.append(tensor_clip)
						
				tensor_clip = torch.cat(tmp_repo, 0)

		return Path(vidpth).name, tensor_clip, label
		

def load_annot_txt(list_txt, use_subset):
	vidpth_framecnt_label = []
	with open(list_txt, "r") as f:
		lines = f.readlines()
	for ii, line in enumerate(lines):
		row = line.strip().split(" ")
		assert len(row) == 3
		vidpth, framecnt, label = row[0], int(row[1]), int(row[2])
		# We "define" the subset of original set to be
		# video where ii%4!=0
		if use_subset and ii % 10 !=0:
			continue
		vidpth_framecnt_label.append([vidpth, framecnt, label])
	return vidpth_framecnt_label


def read_random_clip(frame_vid_folder, frame_cnt, 
				frame_format, t_size, t_step):	
	# Generate a random value
	min_bound = 0
	max_bound = max(frame_cnt - t_size * t_step + 1, 1)
	
	st_idx    = 0 if max_bound == 1 else np.random.randint(0, max_bound -1)
	idxs = [ (idx * t_step + st_idx) % frame_cnt  + 1 for idx in range(t_size) ]
	pil_clip = []
	for idx in idxs:
		img_pth = frame_vid_folder / frame_format.format(idx)
		#print(img_pth)
		pil_img = Image.open(str(img_pth)).convert('RGB')
		pil_clip.append(pil_img)

	return pil_clip


def read_clip(frame_vid_folder, frame_cnt, 
				frame_format, t_size, t_step, sample_times):	
	max_bound = max(frame_cnt - t_size * t_step + 1, 1)
	tick = 1.0 * frame_cnt / sample_times
	if tick == 0:
		tick == 1
	
	st_idxs = np.array([ int(x * tick) for x in range(sample_times) ])
	st_idxs = np.clip(st_idxs, 0, max_bound)

	pil_clip_repo = []
	for st_idx in st_idxs:
		st_idx = 0 if max_bound == 1 else st_idx
		idxs = [ (idx * t_step + st_idx) % frame_cnt  + 1 for idx in range(t_size) ]
		#print(idxs)
		pil_clip = []
		for idx in idxs:
			img_pth = frame_vid_folder / frame_format.format(idx)
			pil_img = Image.open(str(img_pth)).convert('RGB')
			pil_clip.append(pil_img)
		pil_clip_repo.append(pil_clip)

	return pil_clip_repo


def read_random_uniform_clip(frame_vid_folder, frame_cnt, 
				frame_format, t_size):	

	average_druation = frame_cnt // t_size
	if average_druation > 0:
		idxs = np.multiply(list(range(t_size)), average_druation) + np.random.randint(average_druation, size=t_size)
	elif frame_cnt > t_size:
		idxs = np.sort(np.random.randint(frame_cnt + 1, size=t_size))
	else:
		idxs = np.zeros((frame_cnt,))

	idxs = idxs + 1
	# Generate a random value
	pil_clip = []
	for idx in idxs:
		img_pth = frame_vid_folder / frame_format.format(idx)
		pil_img = Image.open(str(img_pth)).convert('RGB')
		pil_clip.append(pil_img)

	return pil_clip


def read_uniform_clip(frame_vid_folder, frame_cnt, 
		    frame_format, t_size, uniform_twice):
	pil_clip_repo = []
	# First
	tick = 1.0 * frame_cnt / t_size
	idxs = np.array(
			[int(tick / 2.0 + tick * x) for x in range(t_size)]
			)
	pil_clip = []
	for idx in idxs:
		img_pth = frame_vid_folder / frame_format.format(idx)
		pil_img = Image.open(str(img_pth)).convert('RGB')
		pil_clip.append(pil_img)
	pil_clip_repo.append(pil_clip)

	# Second
	if uniform_twice:
		pil_clip = []
		idxs2 = np.array(
                [int(tick * x) for x in range(t_size)]
                )
		idxs2 = idxs2 + 1
		for idx2 in idxs2:
			img_pth = frame_vid_folder / frame_format.format(idx)
			pil_img = Image.open(str(img_pth)).convert('RGB')
			pil_clip.append(pil_img)
		pil_clip_repo.append(pil_clip)

	return pil_clip_repo


def data_collate(batch):
	'''
	Custom collate function to deal with batch with non-tensore value
	'''
	batch_vid   = []
	batch_clip  = []
	batch_label = []  

	for sample in batch:
		batch_vid.append(sample[0])
		batch_clip.append(sample[1])
		batch_label.append(sample[2])
	batch_clip = torch.stack(batch_clip, 0)
	return batch_vid, batch_clip, batch_label




