#!/home/hzhang57/anaconda3/envs/cuda10/bin/python
import os
## 1. ViT_224_8x32 Evaluate
#cmd = "/home/hzhang57/anaconda3/envs/cuda10/bin/python -u main_ddp_shift_v3.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#		--evaluate \
#		--resume model_zoo/ViT-B_16_k400_dense_cls400_segs8x32_e18_lr0.1_B21_VAL224/best_vit_B8x32x224_k400.pth \
#		--cfg config/custom/kinetics400/k400_vit_8x32_224.yml"
#os.system(cmd)

### 2. TokShift-Base-16_224_8x32 Evaluate
#cmd = "/home/hzhang57/anaconda3/envs/cuda10/bin/python -u main_ddp_shift_v3.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#		--evaluate \
#		--resume model_zoo/TokShift_ViT-B_16_k400_dense_cls400_segs8x32_e18_lr0.1_B21_VAL224/best_tokshift_B8x32x224_k400.pth \
#		--cfg config/custom/kinetics400/k400_tokshift_div4_8x32_base_224.yml"
#os.system(cmd)
#
## 3. TokShift-Base-16_256_8x32 Evaluate
#cmd = "/home/hzhang57/anaconda3/envs/cuda10/bin/python -u main_ddp_shift_v3.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#		--evaluate \
#		--resume model_zoo/TokShift_ViT-B_16_k400_dense_cls400_segs8x32_e18_lr0.076_B16_VAL256/best_tokshift_B8x32x256_k400.pth \
#		--cfg config/custom/kinetics400/k400_tokshift_div4_8x32_base_256.yml"
#os.system(cmd)

### 4. TokShift-Base-16_224_16x32 Evaluate
#cmd = "/home/hzhang57/anaconda3/envs/cuda10/bin/python -u main_ddp_shift_v3.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#		--evaluate \
#		--resume model_zoo/TokShift_ViT-B_16_k400_dense_cls400_segs16x32_e18_lr0.03_B3_VAL_224/best_tokshift_B16x32x224_k400.pth \
#		--cfg config/custom/kinetics400/k400_tokshift_div4_16x32_base_224.yml"
#os.system(cmd)
#
## 5. TokShift-Large-16_384_8x32 Evaluate
cmd = "/home/hzhang57/anaconda3/envs/cuda10/bin/python -u main_ddp_shift_v3.py \
		--multiprocessing-distributed --world-size 1 --rank 0 \
		--dist-ur tcp://127.0.0.1:23677 \
		--evaluate \
		--resume model_zoo/TokShift_ViT-L_16_k400_dense_cls400_segs8x32_e18_lr0.02_B1_VAL384/best_tokshift_L8x32_k400x384.pth \
		--cfg config/custom/kinetics400/k400_tokshift_div4_8x32_large_384.yml"
os.system(cmd)
#
### 6. TokShift-Large-16_384_12x32 Evaluate
#cmd = "/home/hzhang57/anaconda3/envs/cuda10/bin/python -u main_ddp_shift_v3.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#		--evaluate \
#		--resume model_zoo/TokShift_ViT-L_16_k400_dense_cls400_segs12x32_e18_lr0.03_B1_VAL384/best_tokshift_L12x32x384_k400.pth \
#		--cfg config/custom/kinetics400/k400_tokshift_div4_12x32_large_384.yml"
#os.system(cmd)
#
