#!/usr/bin/env bash
img=/workspace/dataset/AIM-Train/image
tmp=/workspace/dataset/AIM-Train/perfect_trimap
mat=/workspace/dataset/AIM-Train/alpha
fg=/workspace/dataset/AIM-Train/fg
bg=/workspace/dataset/AIM-Train/bg
val_img=/workspace/dataset/AIM-500/original_png
val_tmp=/workspace/dataset/AIM-500/perfect_trimap
val_mat=/workspace/dataset/AIM-500/mask
val_fg=
val_bg=
val_out=data/val
ckpt=checkpoints
patch_size=320
sample=2000
epoch=30

batch=2

t=True

model=g

lr=1e-6

gpu=0

CUDA_VISIBLE_DEVICES=${gpu} python train.py -dgr -m=t-net --lr=${lr} --model=${model} --batch=${batch}  --tolerance_loss=${t} --img=${img} --trimap=${tmp} --matte=${mat} --fg=${fg} --bg=${bg} --val-fg=${val_fg} --val-bg=${val_bg} --val-out=${val_out} --val-img=${val_img} --val-trimap=${val_tmp} --val-matte=${val_mat} --ckpt=${ckpt} --patch-size=${patch_size} --sample=${sample} --epoch=${epoch}
