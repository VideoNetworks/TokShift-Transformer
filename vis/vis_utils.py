from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from torchvision import transforms
import numpy.random as random

def tensor_vis_as_img(tensor_clip, batch_vid):
	batch, t_size, c, h, w = tensor_clip.shape
	for b in range(batch):
		vid_name = batch_vid[b]
		for ii, sig_tensor in enumerate(tensor_clip[b]):
			frame_name = "{}_{}.jpg".format(vid_name, ii)
			sig_tensor = sig_tensor.squeeze()
			image = transforms.ToPILImage()(sig_tensor.cpu()).convert('RGB')
			save_name = "./temp/{}".format(frame_name)
			image.save(save_name, "JPEG")
