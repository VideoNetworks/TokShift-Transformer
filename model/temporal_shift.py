# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers.activations import Swish
from collections import OrderedDict
from utils.raft_utils import coords_grid
import seaborn
import numpy as np
import matplotlib.pyplot as plt
from model.correlation_torch import CorrTorch
from correlation_package.correlation import Correlation

class InplaceShift(torch.autograd.Function):
	# Special thanks to @raoyongming for the help to this function
	@staticmethod
	def forward(ctx, input, fold):
		# not support higher order gradient
		# input = input.detach_()
		ctx.fold_ = fold
		n, t, c, h, w = input.size()
		buffer = input.data.new(n, t, fold, h, w).zero_()
		buffer[:, :-1] = input.data[:, 1:, :fold]
		input.data[:, :, :fold] = buffer
		buffer.zero_()
		buffer[:, 1:] = input.data[:, :-1, fold: 2 * fold]
		input.data[:, :, fold: 2 * fold] = buffer
		return input

	@staticmethod
	def backward(ctx, grad_output):
		# grad_output = grad_output.detach_()
		fold = ctx.fold_
		n, t, c, h, w = grad_output.size()
		buffer = grad_output.data.new(n, t, fold, h, w).zero_()
		buffer[:, 1:] = grad_output.data[:, :-1, :fold]
		grad_output.data[:, :, :fold] = buffer
		buffer.zero_()
		buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
		grad_output.data[:, :, fold: 2 * fold] = buffer
		return grad_output, None


class TemporalShift(nn.Module):
	def __init__(self, net, n_segment=3, n_div=8, inplace=False):
		super(TemporalShift, self).__init__()
		self.net = net
		self.n_segment = n_segment
		self.fold_div = n_div
		self.inplace = inplace
		if inplace:
			print('=> Using in-place shift...')
		print('=> Using fold div: {}'.format(self.fold_div))

	def forward(self, x):
		x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
		return self.net(x)

	@staticmethod
	def shift(x, n_segment, fold_div=3, inplace=False):
		nt, c, h, w = x.size()
		n_batch = nt // n_segment
		x = x.view(n_batch, n_segment, c, h, w)

		fold = c // fold_div
		if inplace:
			# Due to some out of order error when performing parallel computing. 
			# May need to write a CUDA kernel.
			#raise NotImplementedError  
			out = InplaceShift.apply(x, fold)
		else:
			out = torch.zeros_like(x)
			out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
			out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
			out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

		return out.view(nt, c, h, w)


class TemporalShiftV2(nn.Module):
	def __init__(self, net, n_segment=3, n_div=8, inplace=False):
		super(TemporalShiftV2, self).__init__()
		self.net = net
		self.n_segment = n_segment
		self.fold_div = n_div
		self.inplace = inplace
		if inplace:
			print('=> Using in-place shift...')
		print('=> Using fold div: {}'.format(self.fold_div))

	def forward(self, x):
		x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
		return self.net(x)

	@staticmethod
	def shift(x, n_segment, fold_div=3, inplace=False):
		nt, c, h, w = x.size()
		n_batch = nt // n_segment
		x = x.view(n_batch, n_segment, c, h, w)

		fold = c // fold_div
		if inplace:
			# Due to some out of order error when performing parallel computing. 
			# May need to write a CUDA kernel.
			#raise NotImplementedError  
			out = InplaceShift.apply(x, fold)
		else:
			out = torch.zeros_like(x)
			out[:, :-1, :fold] = 0.5 * (x[:, 1:, :fold] + x[:, :-1, :fold]) # shift left
			out[:, 1:, fold: 2 * fold] = 0.5 * (x[:, :-1, fold: 2 * fold] + x[:, 1:, fold:2*fold]) # shift right
			out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

		return out.view(nt, c, h, w)


class TemporalCorr(nn.Module):
	def __init__(self, net, n_segment=3, n_div=8, disp=4, inplace=False):
		super(TemporalShiftV2, self).__init__()
		self.net = net
		self.n_segment = n_segment
		self.fold_div = n_div
		self.inplace = inplace
		if inplace:
			print('=> Using in-place shift...')
		print('=> Using fold div: {}'.format(self.fold_div))
		self.corr_cuda = Correlation(pad_size=disp, kernel_size=1, max_displacement=disp,                                            stride1=1, stride2=2, corr_multiply=1)

	def forward(self, x):
		x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
		return self.net(x)

	def shift(self, x, n_segment, fold_div=3, inplace=False):
		nt, c, h, w = x.size()
		n_batch = nt // n_segment
		x = x.view(n_batch, n_segment, c, h, w)

		fold = c // fold_div

		out = torch.zeros_like(x)
		flow1 = self.corr_cuda(
			x[:, 1:,  :fold].reshape(-1, 1 , h, w),
			x[:, :-1, :fold].reshape(-1, 1, h, w)
		)#ntc, 1, h, w
		flow1 = torch.sqrt(flow1+10e-20)
		flow1 = flow1.view(n*(t-1), c, -1, h, w).mean(2)
		out[:, :-1, :fold] = flow1 # shift left

		flow2 = self.corr_cuda(
			x[:, 1:,  fold:2*fold].view(-1, 1 , h, w),
			x[:, :-1, fold:2*fold].view(-1, 1, h, w)
		)#ntc, 1, h, w
		flow2 = torch.sqrt(flow2+10e-20)
		flow2 = flow2.view(n*(t-1), c, -1, h, w).mean(2)
		out[:, 1:, fold:2*fold] = flow2 # shift right
		out[:, :, 2*fold:] = x[:, :, 2*fold:]  # not shift

		return out.view(nt, c, h, w)


class TemporalConv(nn.Module):
	def __init__(self, net, n_segment=3):
		super(TemporalConv, self).__init__()
		self.net = net
		in_plane = net.in_channels
		#rd = net.out_channels
		rd = in_plane // 8
		self.n_segment = n_segment
		print('=> Using temporal conv: {}'.format(self.n_segment))

		self.squeeze = nn.Sequential(
			nn.Conv2d(in_plane, rd, 1, bias=False),
			nn.BatchNorm2d(rd, eps=1e-05, momentum=0.01),
		)
		self.act = nn.ReLU(inplace=True)

		self.t_conv = nn.Sequential(
			nn.Conv3d(rd, rd, (3, 1, 1), groups=rd, stride=(1,1,1), padding=(1,0,0), bias=False),
			nn.BatchNorm3d(rd, eps=1e-05, momentum=0.01),
		)

		self.expand = nn.Sequential(
			nn.Conv2d(rd, in_plane, 1, bias=False),
			nn.BatchNorm2d(in_plane, eps=1e-05, momentum=0.01),
		)
		
		## Init
		for m in self.t_conv.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
				if m.bias is not None:
					torch.nn.init.uniform_(m.bias)   
				torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

			if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		for m in self.t_conv.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
				if m.bias is not None:
					torch.nn.init.uniform_(m.bias)   
				torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

			if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)


	def forward(self, x):
		nt, c, h, w = x.size()
		n_segment = self.n_segment
		n_batch = nt // n_segment
		# Squeeze
		x = self.squeeze(x)
		x = self.act(x)
		# 3 x 1 x 1
		x = x.view(n_batch, n_segment, -1, h, w)
		x = x.permute(0, 2, 1, 3, 4).contiguous() # b, c, t, h, w
		x = self.t_conv(x)                        # b, c, t, h, w
		x = self.act(x)
		x = x.permute(0, 2, 1, 3, 4)              # b, t, c, h, w
		x = x.reshape(nt, -1, h, w)
		# Expand
		x = self.expand(x)
		x = self.act(x)
		return self.net(x)


class TemporalConv2(nn.Module):
	def __init__(self, net, n_segment=3):
		super(TemporalConv2, self).__init__()
		self.net = net
		in_plane = net.in_channels
		#rd = net.out_channels
		rd = in_plane // 16
		self.n_segment = n_segment
		print('=> Using temporal conv: {}'.format(self.n_segment))

		self.act = nn.ReLU(inplace=True)

		self.t_conv = nn.Sequential(
			nn.Conv3d(in_plane, in_plane, (3, 1, 1), groups=in_plane, 
						stride=(1,1,1), padding=(1,0,0), bias=False),
			nn.BatchNorm3d(in_plane, eps=1e-05, momentum=0.01),
		)
		
		for m in self.t_conv.modules():
			if isinstance(m, nn.Conv3d):
				if m.bias is not None:
					torch.nn.init.uniform_(m.bias)   
				torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

			if isinstance(m, nn.BatchNorm3d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		nt, c, h, w = x.size()
		n_segment = self.n_segment
		n_batch = nt // n_segment
		# 3 x 1 x 1
		x = x.view(n_batch, n_segment, -1, h, w)
		x = x.permute(0, 2, 1, 3, 4).contiguous() # b, c, t, h, w
		x = self.t_conv(x)
		x = self.act(x)
		x = x.permute(0, 2, 1, 3, 4)              # b, t, c, h, w
		x = x.reshape(nt, -1, h, w)
		return self.net(x)


def make_temporal_shift(net, n_segment, n_div=8, place='blockres', temporal_pool=False, ver=0):
	if temporal_pool:
		n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
	else:
		n_segment_list = [n_segment] * 4
	assert n_segment_list[-1] > 0
	print('=> n_segment per stage: {}'.format(n_segment_list))

	# Only For ResNet
	if place == 'block':
		def make_block_temporal(stage, this_segment):
			blocks = list(stage.children())
			print('=> Processing stage with {} blocks'.format(len(blocks)))
			for i, b in enumerate(blocks):
				blocks[i] = TemporalShift(b, n_segment=this_segment, n_div=n_div)
			return nn.Sequential(*(blocks))

		net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
		net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
		net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
		net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

	elif 'blockres' in place:
		n_round = 1
		if len(list(net.layer3.children())) >= 23:
			n_round = 2
			print('=> Using n_round {} to insert temporal shift'.format(n_round))

		def make_block_temporal(stage, this_segment, ver):
			blocks = list(stage.children())
			print('=> Processing stage with {} blocks residual'.format(len(blocks)))
			for i, b in enumerate(blocks):
				if i % n_round == 0:
					if ver == 0:
						blocks[i].conv1 = TemporalShift(b.conv1, n_segment=this_segment, n_div=n_div)
					elif ver == 1:
						blocks[i].conv1 = TemporalConv(b.conv1, n_segment=this_segment)
					elif ver == 2:
						blocks[i].conv1 = TemporalConv2(b.conv1, n_segment=this_segment)
					elif ver == 3:
						blocks[i].conv1 = TemporalShiftV2(b.conv1, n_segment=this_segment)
			return nn.Sequential(*blocks)

		net.layer1 = make_block_temporal(net.layer1, n_segment_list[0], ver)
		net.layer2 = make_block_temporal(net.layer2, n_segment_list[1], ver)
		net.layer3 = make_block_temporal(net.layer3, n_segment_list[2], ver)
		net.layer4 = make_block_temporal(net.layer4, n_segment_list[3], ver)


def make_temporal_pool(net, n_segment):
	# Only Support Resnet
	print('=> Injecting nonlocal pooling')
	net.layer2 = TemporalPool(net.layer2, n_segment)


if __name__ == '__main__':
	# test inplace shift v.s. vanilla shift
	tsm1 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=False)
	tsm2 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=True)

	print('=> Testing CPU...')
	# test forward
	with torch.no_grad():
		for i in range(10):
			x = torch.rand(2 * 8, 3, 224, 224)
			y1 = tsm1(x)
			y2 = tsm2(x)
			assert torch.norm(y1 - y2).item() < 1e-5

	# test backward
	with torch.enable_grad():
		for i in range(10):
			x1 = torch.rand(2 * 8, 3, 224, 224)
			x1.requires_grad_()
			x2 = x1.clone()
			y1 = tsm1(x1)
			y2 = tsm2(x2)
			grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
			grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
			assert torch.norm(grad1 - grad2).item() < 1e-5

	print('=> Testing GPU...')
	tsm1.cuda()
	tsm2.cuda()
	# test forward
	with torch.no_grad():
		for i in range(10):
			x = torch.rand(2 * 8, 3, 224, 224).cuda()
			y1 = tsm1(x)
			y2 = tsm2(x)
			assert torch.norm(y1 - y2).item() < 1e-5

	# test backward
	with torch.enable_grad():
		for i in range(10):
			x1 = torch.rand(2 * 8, 3, 224, 224).cuda()
			x1.requires_grad_()
			x2 = x1.clone()
			y1 = tsm1(x1)
			y2 = tsm2(x2)
			grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
			grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
			assert torch.norm(grad1 - grad2).item() < 1e-5
	print('Test passed.')




