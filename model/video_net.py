import torch
import torch.nn as nn
from torch.nn.init import normal_, constant_
from model.basic_ops import ConsensusModule
import numpy as np

import sys
from importlib import import_module
sys.path.append('..')

class VideoNet(nn.Module):
	def __init__(self, num_class, num_segments, modality,
				backbone='ViT-B_16', net=None, consensus_type='avg',
				dropout=0.5, partial_bn=True, print_spec=True, pretrain='imagenet',
				is_shift=False, shift_div=8,
			        drop_block=0, vit_img_size=224,
				vit_pretrain="", LayerNormFreeze=2, cfg=None):
		super(VideoNet, self).__init__()
		self.num_segments = num_segments
		self.modality = modality
		self.backbone = backbone
		self.net = net
		self.dropout = dropout
		self.pretrain = pretrain
		self.consensus_type = consensus_type
		self.drop_block = drop_block
		self.init_crop_size = 256
		self.vit_img_size=vit_img_size
		self.vit_pretrain=vit_pretrain

		self.is_shift = is_shift
		self.shift_div = shift_div
		self.backbone = backbone
		
		self.num_class = num_class
		self.cfg = cfg
		self._prepare_base_model(backbone)
		if "resnet" in self.backbone:
			self._prepare_fc(num_class)
		self.consensus = ConsensusModule(consensus_type)
		#self.softmax = nn.Softmax()
		self._enable_pbn = partial_bn
		self.LayerNormFreeze = LayerNormFreeze
		if partial_bn:
			self.partialBN(True)

	def _prepare_base_model(self, backbone):
		if 'ViT' in backbone:
			if self.net == 'ViT':
				print('=> base model: ViT, with backbone: {}'.format(backbone))
				from vit_models.modeling import VisionTransformer, CONFIGS
				vit_cfg = CONFIGS[backbone]
				self.base_model = VisionTransformer(vit_cfg, self.vit_img_size,
									zero_head=True, num_classes=self.num_class)
			elif self.net == 'TokShift':
				print('=> base model: TokShift, with backbone: {}'.format(backbone))
				from vit_models.modeling_tokshift import VisionTransformer, CONFIGS
				vit_cfg = CONFIGS[backbone]
				vit_cfg.n_seg = self.num_segments
				vit_cfg.fold_div = self.shift_div
				self.base_model = VisionTransformer(vit_cfg, self.vit_img_size,
									zero_head=True, num_classes=self.num_class)
			if self.vit_pretrain != "":
				print("ViT pretrain weights: {}".format(self.vit_pretrain))
				self.base_model.load_from(np.load(self.vit_pretrain))
			self.feature_dim=self.num_class

		else:
			raise ValueError('Unknown backbone: {}'.format(backbone))


	def _prepare_fc(self, num_class):
		if self.dropout == 0:
			setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(self.feature_dim, num_class))
			self.new_fc = None
		else:
			setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
			self.new_fc = nn.Linear(self.feature_dim, num_class)

		std = 0.001
		if self.new_fc is None:
			normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
			constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
		else:
			if hasattr(self.new_fc, 'weight'):
				normal_(self.new_fc.weight, 0, std)
				constant_(self.new_fc.bias, 0)

	#
	def train(self, mode=True):
		# Override the default train() to freeze the BN parameters
		super(VideoNet, self).train(mode)
		count = 0
		if self._enable_pbn and mode:
			print("Freezing LayerNorm.")
			for m in self.base_model.modules():
				if isinstance(m, nn.LayerNorm):
					count += 1
					if count >= (self.LayerNormFreeze if self._enable_pbn else 1):
						m.eval()
						print("Freeze {}".format(m))
						# shutdown update in frozen mode
						m.weight.requires_grad = False
						m.bias.requires_grad = False


	#
	def partialBN(self, enable):
		self._enable_pbn = enable


	def forward(self, input):
		# input size [batch_size, num_segments, 3, h, w]
		input = input.view((-1, 3) + input.size()[-2:])
		if 'ViT' in self.backbone:
			base_out, atten = self.base_model(input)
			base_out = base_out.view((-1,self.num_segments)+base_out.size()[1:])
			#print("Baseout {}".format(base_out.shape))
			#print(base_out[0,:,1:10])
		#
		output = self.consensus(base_out)
		return output.squeeze(1)

