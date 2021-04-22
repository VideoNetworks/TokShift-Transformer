#!/home/hzhang57/anaconda3/envs/cuda10/bin/python
import os

cmd = "/home/hzhang57/anaconda3/envs/cuda10/bin/python -u main_ddp_shift_v3.py \
		--multiprocessing-distributed --world-size 1 --rank 0 \
		--dist-ur tcp://127.0.0.1:23677 \
		--resume checkpoints/Swin_swin_base_patch4_window7_224_in22k_k400_dense_cls400_segs8x32_e18_lr0.1_gd1.0_ShiftDiv0_bch15_VAL224/ckpt_e7.pth \
		--cfg config/custom/kinetics400/k400_swin_8x32_224.yml"
os.system(cmd)

