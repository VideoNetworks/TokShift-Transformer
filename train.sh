#!/home/hzhang57/anaconda3/envs/cuda10/bin/python
import os

cmd = "/home/hzhang57/anaconda3/envs/cuda10/bin/python -u main_ddp_shift_v3.py \
		--multiprocessing-distributed --world-size 1 --rank 0 \
		--dist-ur tcp://127.0.0.1:23677 \
		--tune_from pretrain/ViT-L_16_Img21.npz \
		--cfg config/custom/kinetics400/k400_tokshift_div4_12x32_large_384.yml"
os.system(cmd)

