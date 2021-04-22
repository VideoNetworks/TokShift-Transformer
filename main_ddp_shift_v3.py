import os
import argparse
import random
import warnings
import numpy as np
from pathlib import Path
import time
import shutil
#import wandb


# Torch
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import MultiStepLR
# myrepo
from model.video_net import VideoNet 
from config.config_utils import load_config	   
from data.augmentation import Base_ClipAug, Train_ClipAug, Some_Train_ClipAug, Test_ClipAug
from data.dataset import DataRepo, data_collate
from utils.utils import AverageMeter, accuracy 
from utils.utils import reduce_tensor

from tensorboardX import SummaryWriter


best_acc1 = 0
best_epoch = -1
def get_args():
	parser = argparse.ArgumentParser(description="PyTorch ViST")
	parser.add_argument('--seed', default=None, type=int,
						help='seed for initializing training.')
	parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
	parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
						help='number of data loading workers (default: 4)')
	parser.add_argument('--gpu', default=None, type=int,
					help='GPU id to use.')
	parser.add_argument('--rank', default=-1, type=int,
					help='node rank for distributed training')
	parser.add_argument('--dist-url', default='tcp://127.0.0.1:33456', type=str,
					help='url used to set up distributed training')
	parser.add_argument('--world-size', default=-1, type=int,
					help='number of nodes for distributed training')
	parser.add_argument('--multiprocessing-distributed', action='store_true',
					help='Use multi-processing distributed training to launch '
						 'N processes per node, which has N GPUs. This is the '
						 'fastest way to use PyTorch for either single node or '
						 'multi node data parallel training')
	parser.add_argument('--dist-backend', default='nccl', type=str,
					help='distributed backend')
	parser.add_argument("--cfg", type=str, 
						default="./config/custom/ssv1_vit_dense_config.yml")
	parser.add_argument('--tune_from', type=str, 
						default="", help='fine-tune from checkpoint')
	parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
	parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')
	parser.add_argument('--print-freq', '-p', default=10, type=int, 
		metavar='N', help='print frequency (default: 10)')
	

	args = parser.parse_args()
	return args


class LabelSmoothingCrossEntropy(nn.Module):
	""" 
	NLL loss with label smoothing.
	"""
	def __init__(self, smoothing=0.1):
		"""
		Constructor for the LabelSmoothing module.
		:param smoothing: label smoothing factor
		"""
		super(LabelSmoothingCrossEntropy, self).__init__()								
		assert smoothing < 1.0 
		self.smoothing = smoothing
		self.confidence = 1. - smoothing

	def forward(self, x, target):
		logprobs = F.log_softmax(x, dim=-1)
		nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
		nll_loss = nll_loss.squeeze(1)
		smooth_loss = -logprobs.mean(dim=-1)
		loss = self.confidence * nll_loss + self.smoothing * smooth_loss
		return loss.mean()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def main():
	args = get_args()
	if args.seed is not None:
		random.seed(args.seed)		
		torch.manual_seed(args.seed)
		cudnn.deterministic = True
		warnings.warn('You have chosen to seed training. '
					  'This will turn on the CUDNN deterministic setting, '
					  'which can slow down your training considerably! '
					  'You may see unexpected behavior when restarting '
					  'from checkpoints.')

	if args.gpu is not None:
		warnings.warn('You have chosen a specific GPU. This will completely '
					  'disable data parallelism.')

	if args.dist_url == "env://" and args.world_size == -1:
		args.world_size = int(os.environ["WORLD_SIZE"])

	args.distributed = args.world_size > 1 or args.multiprocessing_distributed
	ngpus_per_node = torch.cuda.device_count()
	if args.distributed:
		print("Using Distribued mode with world-size {}, {} gpus/node".format(args.world_size,
					ngpus_per_node))
	if args.multiprocessing_distributed:
		# Since we have ngpus_per_node processes per node, the total world_zie
		# needs to be adjusted accordingly
		args.world_size= ngpus_per_node * args.world_size
		# Use torch.multiprocessing.spawn to launch distributed process: the
		# main_worker process function
		mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
	else:
		# Simply call main_worker function
		main_worker(args.gpu, ngpus_per_node, args)

	print("Process Finished")


def main_worker(gpu, ngpus_per_node, args):
	global best_acc1
	global best_epoch
	args.gpu = gpu
	cfg  = load_config(args.cfg)
	if args.gpu is not None:
		print("Use GPU: {} for training".format(args.gpu))

	if args.distributed:
		if args.dist_url == "env://" and args.rank == -1:
			args.rank = int(os.environ["RANK"])
		if args.multiprocessing_distributed:
			# For multiprocessing distributed training, rank needs to be the
			# global rank among all the processes
			args.rank = args.rank * ngpus_per_node + gpu
		dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
			world_size=args.world_size, rank=args.rank)

	# create model
	model = VideoNet(cfg['NUM_CLASS'], cfg['T_SIZE'], 'RGB',
				cfg['BASE_NET'], cfg['NET'],
				cfg['CONSENSUS_TYPE'],
				cfg['DROP_OUT'], cfg['PARTIAL_BN'],
				cfg['PRINT_SPEC'], cfg['PRETRAIN'],
				cfg['IS_SHIFT'], cfg['SHIFT_DIV'],
				cfg['DROP_BLOCK'],
				cfg['VAL_PATCH_SIZE'], args.tune_from, cfg=cfg)
	print(model)

	if not torch.cuda.is_available():
		print('using CPU, this will be slow')
	elif args.distributed:
		# For multiprocessing distributed, DistributedDataParallel constr
		# should always set the single device scope, otherwise,
		# DistributedDataParallel will use all available devices.
		if args.gpu is not None:
			print('Mode1')
			torch.cuda.set_device(args.gpu)
			model.cuda(args.gpu)
			# When using a single GPU per process and per
			# DistributedDataParallel, we need to divide the batch size
			# ourselves based on the total number of GPUs we have
			#batch_size = int(cfg['TRN_BATCH'] / ngpus_per_node) 
			#workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
			#print("gpuBatch {}, gpuWorker {}".format(batch_size, workers))
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
		else:
			print('Mode2')
			model.cuda()
			# DistributedDataParallel will divide and allocate batch_size to all
			# available GPUs if device_ids are not set
			model = torch.nn.parallel.DistributedDataParallel(model)
	elif args.gpu is not None:
		torch.cuda.set_device(args.gpu)
		model = model.cuda(args.gpu)

	# define loss function
	if cfg['LABEL_SMOOTH']:
		criterion = LabelSmoothingCrossEntropy().cuda(args.gpu)
	else:
		criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)

	# define optimizer
	policies = model.parameters()
	cfg['WEIGHT_DECAY'] = float(cfg['WEIGHT_DECAY'])
	optimizer = torch.optim.SGD(policies, 
				    cfg['LR'],
				    momentum=cfg['MOMENTUM'],
				    weight_decay=cfg['WEIGHT_DECAY'])
	scheduler = MultiStepLR(optimizer, milestones=cfg['LR_STEPS']
				,gamma=0.1)


	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			if args.gpu is None:
				checkpoint = torch.load(args.resume)
			else:
				# Map model to be loaded to specified single gpu.
				loc = 'cuda:{}'.format(args.gpu)
				checkpoint = torch.load(args.resume, map_location=loc)
			args.start_epoch = checkpoint['epoch']
			best_acc1 = checkpoint['best_acc1']
			#if args.gpu is not None:
			#	# best_acc1 may be from a checkpoint from a different GPU
			#	best_acc1 = best_acc1.to(args.gpu)
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			#scheduler.load_state_dict(checkpoint['scheduler'])
			scheduler.step(args.start_epoch)
			print("=> loaded checkpoint '{}' (epoch {})"
				.format(args.resume, checkpoint['epoch']))
			del checkpoint

		else:
			print("=> no checkpoint found at '{}'".format(args.resume))
	cudnn.benchmark = True
	pixel_mean = cfg['PIXEL_MEAN']
	pixel_std = cfg['PIXEL_STD']

	# data-loading code
	test_aug_func = Test_ClipAug(cfg['VAL_PATCH_SIZE'],
								cfg['VAL_SHORT_SIDE'],
								pixel_mean=pixel_mean,
								pixel_std=pixel_std,
								mode=cfg['VAL_TEST_AUG'])
	#trn_aug_func  = Some_Train_ClipAug(cfg['TRN_PATCH_SIZE'],
	trn_aug_func  = Train_ClipAug(cfg['TRN_PATCH_SIZE'],
								cfg['TRN_SHORT_SIDE_RANGE'],
								pixel_mean=pixel_mean,
								pixel_std=pixel_std,
								)

	if args.evaluate:
		cfg["VAL_SAMPLE_TIMES"] = 10
	trn_data = DataRepo(cfg, is_train='train', aug=trn_aug_func)
	val_data = DataRepo(cfg, is_train='val', aug=test_aug_func)
	print("Amounts of Train/Validate = {}/{}".format(len(trn_data),
													len(val_data)))
	if args.distributed:
		trn_sampler = torch.utils.data.distributed.DistributedSampler(trn_data)
		val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=False)
	else:								
		trn_sampler = None
		val_sampler = None
	trn_loader = data.DataLoader(trn_data, cfg['TRN_BATCH'],
				    num_workers=args.workers, shuffle=(trn_sampler is None),
				    collate_fn=data_collate, pin_memory=True,
				    sampler=trn_sampler, drop_last=True)
	val_loader = data.DataLoader(val_data, cfg['VAL_BATCH'],
				    num_workers=args.workers, shuffle=(val_sampler is False),
				    collate_fn=data_collate, pin_memory=True,
				    sampler=val_sampler)
	if args.evaluate:
		cfg['epoch_lr'] = optimizer.param_groups[0]['lr']
		validate(val_loader, model, criterion, optimizer, args.start_epoch-1, args, cfg, None)
		return

	store_name = '_'.join(
	[
	#"D{}".format(time.strftime('%Y-%m-%d-%H',time.localtime(time.time()))),
	cfg['NET'],
	cfg['BASE_NET'],
	cfg['DATASET'],
	#'ver{}'.format(cfg['VER']),
	cfg['DENSE_OR_UNIFORM'],
	'cls{}'.format(cfg['NUM_CLASS']),
	'segs{}x{}'.format(cfg['T_SIZE'], cfg['T_STEP']),
	'e{}'.format(cfg['EPOCH']),
	'lr{}'.format(cfg['LR']),
	'gd{}'.format(cfg['CLIP_GD']),
	#"Shift{}".format(cfg['IS_SHIFT']),
	"ShiftDiv{}".format(cfg['SHIFT_DIV']),
	"bch{}".format(cfg['TRN_BATCH']),
	"VAL{}".format(cfg['VAL_PATCH_SIZE']),
	#'LableSmt{}'.format(cfg['LABEL_SMOOTH']),
	#'Mean0.5',
	]
	)
	#wandb.init(project="{}".format(store_name), config={
        #"NET": cfg['NET'],
        #"BASE_NET": cfg["BASE_NET"],
        #"DATASET": cfg['DATASET'],
        #"DENSE_OR_UNIFORM": cfg["DENSE_OR_UNIFORM"],
        #"CLASS": cfg["NUM_CLASS"],
        #"T_SIZE": cfg["T_SIZE"],
        #"T_STEP": cfg["T_STEP"],
        #"Epoch": cfg["EPOCH"],
        #"Lr": cfg["LR"],
        #"ShiftDiv": cfg["SHIFT_DIV"],
        #"Batch":  cfg["TRN_BATCH"],
        #"VAL": cfg["VAL_PATCH_SIZE"],
        #})
        #wandb_cfg = wandb.config

	tf_writer = None
	print(store_name)
	if args.rank == 0:
		tf_writer = SummaryWriter(log_dir=os.path.join("./LOG/", store_name))
		checkpoint_folder = Path('checkpoints') / store_name
		checkpoint_folder.mkdir(parents=True, exist_ok=True)

	for epoch in range(args.start_epoch, cfg['EPOCH']):
		if args.distributed:
			trn_sampler.set_epoch(epoch)

		#cfg['epoch_lr'] = scheduler.get_last_lr()[0]
		cfg['epoch_lr'] = optimizer.param_groups[0]['lr']
		
		# train for one epoch
		train(trn_loader, model, criterion, optimizer, epoch, args, cfg, tf_writer)
		# save checkpoint
		print("Validating ...")
		acc1 = validate(val_loader, model, criterion, optimizer, epoch, args, cfg, tf_writer)
		scheduler.step()
		# remember best acc@1 and save checkpoint
		is_best = acc1 > best_acc1
		best_acc1 = max(acc1, best_acc1)		
		if tf_writer != None and args.rank == 0:
			tf_writer.add_scalar('Accuracy/best_test_top1', best_acc1, epoch)
		if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank==0):
			# save current epoch
			pre_ckpt = checkpoint_folder / "ckpt_e{}.pth".format(epoch-1)
			if os.path.isfile(str(pre_ckpt)):
				os.remove(str(pre_ckpt))
			cur_ckpt = checkpoint_folder / "ckpt_e{}.pth".format(epoch)
			save_checkpoint({
					'epoch': epoch + 1,
					'state_dict': model.state_dict(),
					'best_acc1': best_acc1,
					'optimizer' : optimizer.state_dict(),
					'scheduler' : scheduler.state_dict(),
					}, is_best, filename=cur_ckpt)
		
			if is_best:
				pre_filename = checkpoint_folder / "best_ckpt_e{}.pth".format(best_epoch)
				if os.path.isfile(str(pre_filename)):
					os.remove(str(pre_filename))

				best_epoch = epoch
				filename = checkpoint_folder / "best_ckpt_e{}.pth".format(best_epoch)
				save_checkpoint({
					'epoch': epoch + 1,
					'state_dict': model.state_dict(),
					'best_acc1': best_acc1,
					'optimizer' : optimizer.state_dict(),
					'scheduler' : scheduler.state_dict(),
					}, is_best, filename=filename)
        #wandb.finish()


def train(trn_loader, model, criterion, optimizer, epoch, args, cfg, tf_writer, wandb=None):
	losses     = AverageMeter()
	top1       = AverageMeter()
	top5       = AverageMeter()

	# switch to train mode
	model.train()
	trn_len = len(trn_loader)
	for ii, (batch_vid, batch_clip, batch_label) in enumerate(trn_loader):
		# 1. Data:  Batch x Time x Channel x H x W
		#        Batch x Label
		batch_label = torch.LongTensor(batch_label)
		if args.gpu is not None:
			batch_label = batch_label.cuda(args.gpu, non_blocking=True)
			batch_clip  = batch_clip.cuda(args.gpu, non_blocking=True)
			#print("target {}, clip {}".format(batch_label.shape, batch_clip.shape))
			output = model(batch_clip)
			loss = criterion(output, batch_label)
			prec1, prec5 = accuracy(output.data, batch_label, topk=(1,5)) 
			if cfg['GRADIENT_ACCUMULATION_STEPS'] > 1:
				loss = loss / cfg['GRADIENT_ACCUMULATION_STEPS']
			loss.backward()

			# Gather Display
			if args.distributed:
				reduce_loss = reduce_tensor(args, loss.data * cfg['GRADIENT_ACCUMULATION_STEPS'])
				rprec1 = reduce_tensor(args, prec1)
				rprec5 = reduce_tensor(args, prec5)
			else:
				reduce_loss = loss.detach().data * cfg['GRADIENT_ACCUMULATION_STEPS']
				rprec1 = prec1.detach().data
				rprec5 = prec5.detach().data

			losses.update(reduce_loss.item())
			top1.update(rprec1.item())
			top5.update(rprec5.item())
			

			### Accumultate Backprogate
			if (ii + 1) % cfg['GRADIENT_ACCUMULATION_STEPS'] == 0 or ii == trn_len - 1:
				if cfg['CLIP_GD'] > 0:
					clip_grad_norm_(model.parameters(), cfg['CLIP_GD'])
				optimizer.step()
				optimizer.zero_grad()

				# Display Progress
				if args.rank == 0:
					print("TRN Epoch [{}][{}/{}], lr {:.6f}, loss: {:.4f}, Acc1: {:.3f}, Acc5: {:.3f}".format(epoch, ii, trn_len, cfg['epoch_lr'], losses.avg, top1.avg, top5.avg))
			###-----


	del loss, output, batch_clip, batch_label, prec1, prec5, reduce_loss
	if tf_writer != None and args.rank == 0:
		tf_writer.add_scalar('Loss/train', losses.avg, epoch)
		tf_writer.add_scalar('LearningRate', cfg['epoch_lr'], epoch)
		tf_writer.add_scalar('Accuracy/train_top1', top1.avg, epoch)
		tf_writer.add_scalar('Accuracy/train_top5', top5.avg, epoch)
	#if wandb != None and args.rank == 0:
	#	wandb.log({'epoch': epoch, 'train_loss': losses.avg, 'train_top1': top1.avg, "train_top5": top5.avg})


def validate(val_loader, model, criterion, optimizer, epoch, args, cfg, tf_writer, wandb=None):
	losses     = AverageMeter()
	top1       = AverageMeter()
	top5       = AverageMeter()

	# switch to train mode
	model.eval()
	val_len = len(val_loader)
	with torch.no_grad():
		for ii, (batch_vid, batch_clip, batch_label) in enumerate(val_loader):
			# 1. Data:  Batch x Mode x Time x Channel x H x W
			#        Batch x Label
			batch_label = torch.LongTensor(batch_label)
			if args.gpu is not None:
				batch_label = batch_label.cuda(args.gpu, non_blocking=True)
				batch_clip  = batch_clip.cuda(args.gpu, non_blocking=True)
				Batch, M, T, C, H, W = batch_clip.shape
				#print("target {}, clip {}".format(batch_label.shape, batch_clip.shape))
				batch_clip = batch_clip.view(-1, T, C, H, W)
				batch_label_dup = batch_label.unsqueeze(0).repeat(M,1)
				batch_label_dup = batch_label_dup.permute(1,0).contiguous().view(-1)
				
				output = model(batch_clip)
				loss = criterion(output, batch_label_dup)
				output = output.view(Batch, M, output.size(-1))
				# Average across M
				output = output.view(Batch, M, output.size(-1))
				output = output.mean(1)

				prec1, prec5 = accuracy(output.data, batch_label, topk=(1,5)) 

				# Gather Display
				if args.distributed:
					reduce_loss = reduce_tensor(args, loss.data)
					rprec1 = reduce_tensor(args, prec1)
					rprec5 = reduce_tensor(args, prec5)
				else:
					reduce_loss = loss.detach().data
					rprec1 = prec1.detach().data
					rprec5 = prec5.detach().data

				losses.update(reduce_loss.item(), Batch)
				top1.update(rprec1.item(), Batch)
				top5.update(rprec5.item(), Batch)
				

				### Accumultate Backprogate
				if ii  % args.print_freq == 0 or ii == val_len - 1:
					# Display Progress
					if args.rank == 0:
						print("Val Epoch [{}][{}/{}], lr {:.6f}, loss: {:.4f}, Acc1 {:.3f}, Acc5 {:.3f}".format(epoch, ii, val_len, cfg['epoch_lr'], losses.avg, top1.avg, top5.avg))
				###-----


		del loss, output, batch_clip, batch_label, prec1, prec5, reduce_loss, rprec1, rprec5
		if tf_writer != None and args.rank == 0:
			tf_writer.add_scalar('Loss/test', losses.avg, epoch)
			tf_writer.add_scalar('Accuracy/test_top1', top1.avg, epoch)
			tf_writer.add_scalar('Accuracy/test_top5', top5.avg, epoch)
	
	print("Evaluate Val {}: loss {}, top1 {} ".format(epoch, losses.avg, top1.avg))
	print("--"*10)
	return top1.avg

if __name__ == "__main__":
	main()

