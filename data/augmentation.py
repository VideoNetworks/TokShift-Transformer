import math
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from numpy import random

'''
Compose_func collect several augmentations together
Args: aug_funcs (List[aug_func]): A list of augment functions to compose
Example:
	>>> Compose_func([
	>>>     clip_random_brightness_func,
	>>>     clip_random_hue_func,
	>>>     clip_random_saturation_func
'''
class Compose_func(object):
	def __init__(self, aug_funcs):
		self.aug_funcs = aug_funcs


	def __call__(self, pil_clip, label):
		## pil_clip is List[PIL]
		for aug in self.aug_funcs:
			pil_clip, label = aug(pil_clip, label)

		return pil_clip, label

'''
Random Brightness
'''
class clip_random_brightness(object):
	def __init__(self, prob, brightness=1):
		self.value = [ 0.5, 1+brightness ]
		self.value[0] = max(self.value[0], 0)
		self.prob = prob

	def __call__(self, pil_clip, label):
		if random.randint(self.prob):
			return pil_clip, label
		else:
			brightness = random.uniform(self.value[0], self.value[1])
			pil_clip = [ transforms.functional.adjust_brightness(x, brightness) for x in pil_clip ]
			return pil_clip, label


'''
Random Saturation
'''
class clip_random_saturation(object):
	def __init__(self, prob, saturation=2):
		self.value = [ 0.5, 1+saturation ]
		self.value[0] = max(self.value[0], 0)
		self.prob = prob
		

	def __call__(self, pil_clip, label):
		if random.randint(self.prob):
			return pil_clip, label
		else:
			saturation = random.uniform(self.value[0], self.value[1])
			pil_clip   = [ transforms.functional.adjust_saturation(x, saturation) for x in pil_clip ]
			return pil_clip, label


'''
Random Gamma
'''
class clip_random_gamma(object):
	def __init__(self, prob, gamma=0.2):
		self.value = [ 1-gamma, 1+gamma ]
		self.value[0] = max(self.value[0], 0)
		self.prob = prob
	def __call__(self, pil_clip, label):
		if random.randint(self.prob):
			return pil_clip, label
		else:
			gamma = random.uniform(self.value[0], self.value[1])
			pil_clip = [ transforms.functional.adjust_gamma(x, gamma) for x in pil_clip ]
			return pil_clip, label


'''
Random Hue
'''
class clip_random_hue(object):
	def __init__(self, prob):
		self.prob = prob
		self.value = [-0.5, 0.5]
	def __call__(self, pil_clip, label):
		if random.randint(self.prob):
			return pil_clip, label
		else:
			hue = random.uniform(self.value[0], self.value[1])
			pil_clip = [ transforms.functional.adjust_hue(x, hue) for x in pil_clip ]
			return pil_clip, label


'''
Random Horizontal Flip
'''
class clip_random_hflip(object):
	def __init__(self, prob):
		self.prob = prob # prob to flip

	def __call__(self, pil_clip, label):
		if random.randint(self.prob):
			return pil_clip, label
		else:
			pil_clip = [ transforms.functional.hflip(x) for x in pil_clip]
			return pil_clip, label


'''
Random Horizontal Flip
'''
class some_clip_random_hflip(object):
	def __init__(self, prob):
	        self.prob = prob # prob to flip
	        self.map = {86:87,87:86,93:94,94:93,166:167,167:166}

	def __call__(self, pil_clip, label):
		if random.randint(self.prob):
			return pil_clip, label
		else:
			pil_clip = [ transforms.functional.hflip(x) for x in pil_clip]
			if label in self.map.keys():
                            label = self.map[label]
			return pil_clip, label


'''
Center-Crop a PIL clip
'''
class clip_center_crop(object):
	'''
	Center Crop a 224 x 224 patch from iamges/clips processed by short-side
	'''
	def __init__(self, patch_size=224, short_side=256):
		self.patch_size = patch_size
		self.short_side = short_side

	
	def __call__(self, pil_clip, label):
		# Resize short side to fix
		pil_clip = [ short_side_resize(x, self.short_side) for x in pil_clip ]
		# Center Crop
		width, height = pil_clip[0].size
		y_offset = 0
		if height > self.patch_size:
			y_offset = int( ( height - self.patch_size) / 2.0)
		x_offset = 0
		if width > self.patch_size:
			x_offset = int( ( width - self.patch_size) / 2.0)
		pil_clip = [ image_crop(x, x_offset, y_offset, self.patch_size) for x in pil_clip ]
		return pil_clip, label


'''
Crop a PIL clip
'''
class clip_crop(object):
	'''
	Crop a 224 x 224 patch from iamges/clips processed by short-side
	'''
	def __init__(self, patch_size=224, short_side=256, cp_type=1):
		self.patch_size = patch_size
		self.short_side = short_side
		self.cp_type = cp_type
	
	def __call__(self, pil_clip, label):
		# Resize short side to fix
		pil_clip = [ short_side_resize(x, self.short_side) for x in pil_clip ]
		# Center Crop
		width, height = pil_clip[0].size

		if self.cp_type == 1: # Left-Center Crop
			x_offset = 0
			y_offset = 	int( ( height - self.patch_size) / 2.0)

		if self.cp_type == 2: # Right-Center Crop
			x_offset = width - self.patch_size
			y_offset = 	int( ( height - self.patch_size) / 2.0)
		
		if self.cp_type == 3: # Top-Left Crop
			x_offset = 0
			y_offset = 0
		
		if self.cp_type == 4: # Top-Rigt Crop
			x_offset = width - self.patch_size
			y_offset = 0

		if self.cp_type == 5: # Bot-Left Crop
			x_offset = 0
			y_offset = height - self.patch_size

		if self.cp_type == 6: # Bot-Right Crop
			x_offset = width - self.patch_size
			y_offset = height - self.patch_size

		x_offset = max(x_offset, 0)
		y_offset = max(y_offset, 0)

		pil_clip = [ image_crop(x, x_offset, y_offset, self.patch_size) for x in pil_clip ]
		return pil_clip, label


'''
Random Crop a PIL Clip
'''
class clip_random_crop(object):
	'''
	Random Crop a 224 x 224 patch from images/clips processed by short-side
	'''
	def __init__(self, patch_size=224, short_side_range=[256, 320]):
		self.patch_size = patch_size
		self.short_side_range = short_side_range

	def __call__(self, pil_clip, label):
		# Resize shoft side to fix
		short_side = random.randint(self.short_side_range[0], self.short_side_range[1])
		pil_clip = [ short_side_resize(x, short_side) for x in pil_clip ]
		# Random Crop
		width, height = pil_clip[0].size
		y_offset = 0
		if height > self.patch_size:
			y_offset = int(np.random.randint(0, height - self.patch_size))
		x_offset = 0
		if width > self.patch_size:
			x_offset = int(np.random.randint(0, width - self.patch_size))
		pil_clip = [ image_crop(x, x_offset, y_offset, self.patch_size) for x in pil_clip]

		return pil_clip, label



def image_crop(pil_img, x_offset, y_offset, patch_size):
	width, height = pil_img.size
	left   = x_offset
	right  = x_offset + patch_size
	top    = y_offset
	bottom = y_offset + patch_size 
	
	pil_img = pil_img.crop((left, top, right, bottom))
	return pil_img


'''
Resize Short Side to fix-len, keep original ratio
'''
def short_side_resize(pil_img, short_side):
	width, height = pil_img.size
	if width > height:
		new_height = short_side
		new_width  = int(math.floor(1.0 * width / height * new_height ))
	else:
		new_width  = short_side
		new_height = int(math.floor(1.0 * height / width * new_width ))
	#resized_img = pil_img.resize((new_width, new_height), Image.ANTIALIAS)
	resized_img = pil_img.resize((new_width, new_height), Image.BICUBIC)
	return resized_img


'''
Base Augmentation: Center-Crop, 
	NormalizePixel ValueSubtract Mean, Divide Std
'''
class Base_ClipAug(object):
	def __init__(self, patch_size, short_side,
		pixel_mean=(123, 117, 104), pixel_std=(0.229, 0.224, 0.225)):
		self.pixel_mean = np.array(pixel_mean) / 255.0
		self.pixel_std  = pixel_std

		## 0 Resize Short-Side to Fix-len
		self.clip_resize = clip_center_crop(patch_size = patch_size,
											short_side = short_side)
		## 1. Normalize and to tensor
		self.base_norm = transforms.Compose([
			transforms.ToTensor(), # RGB (0-255) ==> (0-1.0) (PIL in, Tensor out)
			transforms.Normalize(self.pixel_mean, self.pixel_std)
			])


	def __call__(self, pil_clip, label):
		pil_clip, label = self.clip_resize(pil_clip, label)
		tmp_clip    = [ self.base_norm(x) for x in pil_clip ]
		tensor_clip = torch.stack(tmp_clip, dim=0)
		return tensor_clip, label


'''
Test Augmentation: 
	3: Left, Center, Right-Crop, 
	NormalizePixel ValueSubtract Mean, Divide Std
'''
class Test_ClipAug(object):
	def __init__(self, patch_size, short_side, mode=3,
		pixel_mean=(123, 117, 104), pixel_std=(0.229, 0.224, 0.225), 
		memory_aug=0):
		self.pixel_mean = np.array(pixel_mean) / 255.0
		self.pixel_std  = pixel_std
		self.mode = mode
		self.memory_aug = memory_aug		

		## 0 Resize Short-Side to Fix-len
		self.center_crop = clip_center_crop(patch_size = patch_size,
											short_side = short_side)

		self.left_crop = clip_crop(patch_size = patch_size,
									short_side = short_side,
									cp_type=1)
		self.right_crop = clip_crop(patch_size = patch_size,
									short_side = short_side,
									cp_type=2)

		self.topleft_crop = clip_crop(patch_size = patch_size,
									  short_side = short_side,
									cp_type=3)
		self.topright_crop = clip_crop(patch_size = patch_size,
									   short_side = short_side,
									cp_type=4)

		self.botleft_crop = clip_crop(patch_size = patch_size,
									 short_side = short_side,
									cp_type=5)

		self.botright_crop = clip_crop(patch_size = patch_size,
									   short_side = short_side,
									cp_type=6)
		## 1. Normalize and to tensor
		self.base_norm = transforms.Compose([
			transforms.ToTensor(), # RGB (0-255) ==> (0-1.0) (PIL in, Tensor out)
			transforms.Normalize(self.pixel_mean, self.pixel_std)
			])


	def norm_to_tensor(self, pil_clip):
		tmp_clip = [ self.base_norm(x) for x in pil_clip ]
		t_size = len(tmp_clip)
		for ii in range(self.memory_aug, 0, -1):
			if 2* ii < t_size+1:
				tmp_clip.insert(ii-1, tmp_clip[ii-1] )
				tmp_clip.insert(-ii, tmp_clip[-ii])
			else:
				tmp_clip.insert(ii, tmp_clip[ii] )

		tensor_clip = torch.stack(tmp_clip, dim=0)
		return tensor_clip


	def __call__(self, pil_clip, label):

		Test_clips = []
		# Center-Crop
		p0, label   = self.center_crop(pil_clip, label)
		p0 = self.norm_to_tensor(p0)
		Test_clips.append(p0)


		if self.mode == 3:
			# Left-Crop
			p1, label = self.left_crop(pil_clip, label)
			p1 = self.norm_to_tensor(p1)
			Test_clips.append(p1)

			# Right-Crop
			p2, label = self.right_crop(pil_clip, label)
			p2 = self.norm_to_tensor(p2)
			Test_clips.append(p2)

		if self.mode == 5:
			# Top-Left
			p3, label = self.topleft_crop(pil_clip, label)
			p3 = self.norm_to_tensor(p3)
			Test_clips.append(p3)

			# Top-Right
			p4, label = self.topright_crop(pil_clip, label)
			p4 = self.norm_to_tensor(p4)
			Test_clips.append(p4)

			# Bot-Left
			p5, label = self.botleft_crop(pil_clip, label)
			p5 = self.norm_to_tensor(p5)
			Test_clips.append(p5)

			# Bot-Right
			p6, label = self.botright_crop(pil_clip, label)
			p6 = self.norm_to_tensor(p6)
			Test_clips.append(p6)


		Test_clips = torch.stack(Test_clips, dim=0) # Mode x T x C x H x W
		return Test_clips, label


'''
Train Augmentation: Random hfilp, Crop, 
	NormalizePixel ValueSubtract Mean, Divide Std
'''
class Train_ClipAug(object):
	def __init__(self, patch_size, short_side_range,
		pixel_mean=(123, 117, 104), pixel_std=(0.229, 0.224, 0.225), memory_aug=0):
		self.pixel_mean = np.array(pixel_mean) / 255.0
		self.pixel_std  = pixel_std
		self.memory_aug = memory_aug
		## 0 Random Flip, Crop
		aug_list = [ clip_random_hflip(2),
					 clip_random_crop(patch_size = patch_size,
									  short_side_range = short_side_range),
					 clip_random_brightness(10),
					 clip_random_saturation(10),
					 clip_random_gamma(10),
					 clip_random_hue(10),
						]
		self.compose_func = Compose_func(aug_list)
		## 1. Normalize and to tensor
		self.base_norm = transforms.Compose([
			transforms.ToTensor(), # RGB (0-255) ==> (0-1.0) (PIL in, Tensor out)
			transforms.Normalize(self.pixel_mean, self.pixel_std)
			])


	def __call__(self, pil_clip, label):
		pil_clip, label = self.compose_func(pil_clip, label)
		tmp_clip    = [ self.base_norm(x) for x in pil_clip ]
		t_size = len(tmp_clip)
		for ii in range(self.memory_aug, 0, -1):
			if 2* ii < t_size+1:
				tmp_clip.insert(ii-1, tmp_clip[ii-1] )
				tmp_clip.insert(-ii, tmp_clip[-ii])
			else:
				tmp_clip.insert(ii, tmp_clip[ii] )
		
		#for ii in range(self.memory_aug):
		#	tmp_clip.insert(0, frame_interpolate(tmp_clip[0], tmp_clip[1]))
		#	tmp_clip.append(frame_interpolate(tmp_clip[-1], tmp_clip[-2]))

		tensor_clip = torch.stack(tmp_clip, dim=0)
		return tensor_clip, label


'''
SomeSomeTrain Augmentation: Random hfilp, Crop, 
	NormalizePixel ValueSubtract Mean, Divide Std
'''
class Some_Train_ClipAug(object):
	def __init__(self, patch_size, short_side_range,
		pixel_mean=(123, 117, 104), pixel_std=(0.229, 0.224, 0.225), memory_aug=0):
		self.pixel_mean = np.array(pixel_mean) / 255.0
		self.pixel_std  = pixel_std
		self.memory_aug = memory_aug
		## 0 Random Flip, Crop
		aug_list = [ some_clip_random_hflip(2),
					 clip_random_crop(patch_size = patch_size,
									  short_side_range = short_side_range),
					 clip_random_brightness(10),
					 clip_random_saturation(10),
					 clip_random_gamma(10),
					 clip_random_hue(10),
						]
		self.compose_func = Compose_func(aug_list)
		## 1. Normalize and to tensor
		self.base_norm = transforms.Compose([
			transforms.ToTensor(), # RGB (0-255) ==> (0-1.0) (PIL in, Tensor out)
			transforms.Normalize(self.pixel_mean, self.pixel_std)
			])


	def __call__(self, pil_clip, label):
		pil_clip, label = self.compose_func(pil_clip, label)
		tmp_clip    = [ self.base_norm(x) for x in pil_clip ]
		t_size = len(tmp_clip)
		for ii in range(self.memory_aug, 0, -1):
			if 2* ii < t_size+1:
				tmp_clip.insert(ii-1, tmp_clip[ii-1] )
				tmp_clip.insert(-ii, tmp_clip[-ii])
			else:
				tmp_clip.insert(ii, tmp_clip[ii] )
		
		#for ii in range(self.memory_aug):
		#	tmp_clip.insert(0, frame_interpolate(tmp_clip[0], tmp_clip[1]))
		#	tmp_clip.append(frame_interpolate(tmp_clip[-1], tmp_clip[-2]))

		tensor_clip = torch.stack(tmp_clip, dim=0)
		return tensor_clip, label


def frame_interpolate(mid_frame, fb_frame):
	#return 2* mid_frame - fb_frame
	return mid_frame






