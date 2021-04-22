import time
import torch
import functools
import pickle
import logging
import numpy as np
from scipy.sparse import csr_matrix
import torch.distributed as dist           

logger = logging.getLogger(__name__)

@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def get_world_size() -> int:
	if not dist.is_available():
		return 1
	if not dist.is_initialized():
		return 1
	return dist.get_world_size()


def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_tensor(args, tensor):
	rt = tensor.clone()
	dist.all_reduce(rt, op=dist.ReduceOp.SUM)
	rt /= args.world_size
	return rt

def synchronize():
	"""
	Helper function to synchronize (barrier) among all processes when
	using distributed training
	"""
	if not dist.is_available():
		return
	if not dist.is_initialized():
		return
	world_size = dist.get_world_size()
	if world_size == 1:
		return
	dist.barrier()

class MultiLabelBinarizer(object):
	def __init__(self, num_classes):
		self.num_classes = num_classes

	def fit_label(self, labels_batch):
		target = []
		for i, labels in enumerate(labels_batch):
			for j, box_label in enumerate(labels):
				label_arr = self.construct_label_array(box_label)						
				target.append(label_arr)	
		return np.array(target, dtype=np.float)

	def construct_label_array(self, box_label):
		"""Construction label array."""
		label_arr = np.zeros(self.num_classes)
		# AVA label index starts from 1.
		for lbl in box_label:
			if lbl == -1:
				continue
			assert lbl >= 1 and lbl <= 80
			label_arr[lbl - 1] = 1
		return label_arr.astype(np.float)




class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.value = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, value, n=1):
		self.value = value
		self.sum += value * n
		self.count += n
		self.avg = self.sum / self.count


def sec_to_frame(sec, time_offset, fps):
	"""
	Convert time index (in second) to frame index.
	"""
	# Take AVA for example
	# 900 seconds  ---> 0 frames
	# 1801 seconds ---> 27030 frames
	return (sec - time_offset) * fps

def frame_to_sec(frame, time_offset, fps):
	return int(frame) // int(fps) + time_offset



def load_boxes_and_labels(csvfile, valid_frames, is_train, detect_thresh, use_subset):
	"""
	Loading boxes and labels from csv files.
	"""
	ret = {}
	count = 0
	unique_box_count = 0

	with open(csvfile, 'r') as f:
		lines = f.readlines()

	for line in lines:
		row = line.strip().split(',')
		assert len(row) == 7 or len(row) == 8
		video_name, frame_sec = row[0], int(row[1])

		# We "define" the subset of AVA set to be the
		# frames where frame_sec % 4 == 0.
		if (use_subset and frame_sec % 4 != 0):
			continue

		box_key = ','.join(row[2:6])
		box = map(float, row[2:6])
		box = [ x for x in box ]
		label = -1 if row[6] == '' else int(row[6])

		if len(row) == 8:
			# When we use predicted boxes to train/eval, we have scores.
			score = float(row[7])
			if score < detect_thresh:
				continue

		if video_name not in ret:
			ret[video_name] = {}
			for sec in valid_frames:
				ret[video_name][sec] = {}

		if box_key not in ret[video_name][frame_sec]:
			ret[video_name][frame_sec][box_key] = [box, []]
			unique_box_count += 1

		ret[video_name][frame_sec][box_key][1].append(label)
		if label != -1:
			count += 1

	for video_name in ret.keys():
		for frame_sec in ret[video_name].keys():
			dict_values = ret[video_name][frame_sec].values()
			dict_values = [ x for x in dict_values ]
			ret[video_name][frame_sec] = dict_values

	logger.info('Finished loading annotations from')
	logger.info("  {}".format(csvfile))
	logger.info("Number of unique boxes: {}".format(unique_box_count))
	logger.info("Number of annotations: %d".format(count))
	return ret


def gen_datalist(boxes_and_labels, data_configs, valid_frames):
	"""
	A list of frame-kits for Train/Test
	"""
	
	frame_kits = []
	# Getting frame that will be used for training and testing.
	# For trainig, we only need to train on frames with boxes/labels.
	# For testing, we only need to predict for frames with predicted boxes

	count = 0
	for video_name in boxes_and_labels.keys():
		for sec in boxes_and_labels[video_name].keys():
			if sec not in valid_frames:
				logger.info(sec)
				continue

			if len(boxes_and_labels[video_name][sec]) > 0:
				boxlabels = boxes_and_labels[video_name][sec]
				single_frame_kit = gen_kit(video_name, sec, boxlabels, data_configs)
				frame_kits.append(single_frame_kit)

	return frame_kits

 
def gen_kit(video_name, sec, boxlabels, data_configs):
	frame_idx = sec_to_frame(sec, data_configs["TIME_OFFSET"], data_configs["FPS"] )
	start_frame = sec_to_frame(data_configs["VALID_SEC_START"], data_configs["TIME_OFFSET"], data_configs["FPS"]) + 1
	stop_frame  = sec_to_frame(data_configs['VALID_SEC_STOP'], data_configs["TIME_OFFSET"], data_configs["FPS"])
	#gen-warping clips
	m_ahead = data_configs["CLIP_M_FRAMES_AHEAD"]
	n_after = data_configs["CLIP_N_FRAMES_AFTER"]
	step    = data_configs["CLIP_STEP"]
	clip = gen_clip(frame_idx, start_frame, stop_frame, m_ahead, n_after, step)
	boxes, labels = separate_boxes_labels(boxlabels)
	single_kit = [video_name.strip(),
				  frame_idx,
				  clip,
				  boxes, # boxes
				  labels  # labels
				 ]
	# print(single_kit)
	# An example single-kit
	#['8aMv-ZGD4ic', 
	# 25920, 
	# [25913, 25914, 25915, 25916, 25917, 25918, 25919, 25920, 25921, 25922, 25923, 25924, 25925, 25926, 25927, 25928], 
	# [[0.019, 0.108, 0.94, 0.945]], 
	# [[12, 79]]]

	return single_kit
	

def gen_clip(frame_idx, start_frame, stop_frame, m_ahead, n_after, step):
	"""Generating wariping clips"""
	clip_range = np.arange( frame_idx - m_ahead * step, frame_idx + (n_after + 1)*step, step)
	#clip_range = np.arange( frame_idx - m_ahead, frame_idx + (n_after + 1))
	clip_range = np.clip(clip_range, start_frame, stop_frame)

	return clip_range.tolist()


def separate_boxes_labels(boxlabels):
	boxes  = []
	labels = []

	for x in boxlabels:
		boxes.append(x[0])
		labels.append(x[1])
	return boxes, labels


def data_collate(batch):
	"""
	Custom collate function to deal with batchs of images/clips wich have a different number of associated object annotations (bounding boxes).
	"""
	#start_time = time.time()
	batch_video_sec_ids = []
	batch_imgs	= []
	batch_clips   = []
	batch_boxes   = []
	batch_labels = []
	for sample in batch:
		batch_imgs.append(sample[0])
		batch_clips.append(sample[1])
		batch_boxes.append(np.array(sample[2], dtype=np.float16))
		batch_labels.append(sample[3])
		batch_video_sec_ids.append(sample[4])

	batch_imgs = torch.stack(batch_imgs, 0)
	batch_clips = torch.stack(batch_clips, 0)
	#elapsed_time = time.time() - start_time
	#print("A batch take {}s".format(elapsed_time))
	return batch_imgs, batch_clips, batch_boxes, batch_labels, batch_video_sec_ids
