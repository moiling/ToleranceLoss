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
epoch=200
batch=6
t=True
model=g
sample1=2000
sample2=1000
sample3=500
lr1=1e-6
lr2=5e-7
lr3=1e-7
name=tolerance_perfect
gpu=0

mkdir checkpoints_old
mkdir data_old
#-------------------------------------------------------------------------------------
# train: sample=2000, lr=1e-6
CUDA_VISIBLE_DEVICES=${gpu} python train.py -dgr -m=t-net --lr=${lr1} --model=${model} --batch=${batch}  --tolerance_loss=${t} --img=${img} --trimap=${tmp} --matte=${mat} --fg=${fg} --bg=${bg} --val-fg=${val_fg} --val-bg=${val_bg} --val-out=${val_out} --val-img=${val_img} --val-trimap=${val_tmp} --val-matte=${val_mat} --ckpt=${ckpt} --patch-size=${patch_size} --sample=${sample1} --epoch=${epoch}
# save ckpt to local.
# sudo (docker needn't sudo).
cp checkpoints/t-net-best.pt checkpoints_old/${name}-s-${sample1}-lr-${lr1}.pt
# validate:
CUDA_VISIBLE_DEVICES=${gpu} python validate.py -dg -m=t-net --model=${model} --tolerance_loss=${t} --val_img=${val_img} --val_trimap=${val_tmp} --val_matte=${val_mat} --val_fg=${val_fg} --val_bg=${val_bg} --out=${val_out} --ckpt_path=${ckpt}/t-net-best.pt --patch-size=10000
# save data to local.
# sudo
cp -r data/val/trimap/ data_old/${name}-s-${sample1}-lr-${lr1}/
# save date to /share. (docker can't visit /share)
# sudo cp -r data/val/trimap/ /share/data/${name}-s-${sample1}-lr-${lr1}/
# sudo
mv checkpoints/t-net-best.pt checkpoints/${name}-s-${sample1}-lr-${lr1}.pt

#-------------------------------------------------------------------------------------
# train: sample=1000, lr=5e-7
CUDA_VISIBLE_DEVICES=${gpu} python train.py -dgr -m=t-net --lr=${lr2} --model=${model} --batch=${batch}  --tolerance_loss=${t} --img=${img} --trimap=${tmp} --matte=${mat} --fg=${fg} --bg=${bg} --val-fg=${val_fg} --val-bg=${val_bg} --val-out=${val_out} --val-img=${val_img} --val-trimap=${val_tmp} --val-matte=${val_mat} --ckpt=${ckpt} --patch-size=${patch_size} --sample=${sample2} --epoch=${epoch}
# save ckpt to local.
# sudo
cp checkpoints/t-net-best.pt checkpoints_old/${name}-s-${sample2}-lr-${lr2}.pt
# validate:
CUDA_VISIBLE_DEVICES=${gpu} python validate.py -dg -m=t-net --model=${model} --tolerance_loss=${t} --val_img=${val_img} --val_trimap=${val_tmp} --val_matte=${val_mat} --val_fg=${val_fg} --val_bg=${val_bg} --out=${val_out} --ckpt_path=${ckpt}/t-net-best.pt --patch-size=10000
# save data to local.
# sudo
cp -r data/val/trimap/ data_old/${name}-s-${sample2}-lr-${lr2}/
# save date to /share.
# sudo cp -r data/val/trimap/ /share/data/${name}-s-${sample2}-lr-${lr2}/
# sudo
mv checkpoints/t-net-best.pt checkpoints/${name}-s-${sample2}-lr-${lr2}.pt

#-------------------------------------------------------------------------------------
# train: sample=500, lr=1e-7
CUDA_VISIBLE_DEVICES=${gpu} python train.py -dgr -m=t-net --lr=${lr3} --model=${model} --batch=${batch}  --tolerance_loss=${t} --img=${img} --trimap=${tmp} --matte=${mat} --fg=${fg} --bg=${bg} --val-fg=${val_fg} --val-bg=${val_bg} --val-out=${val_out} --val-img=${val_img} --val-trimap=${val_tmp} --val-matte=${val_mat} --ckpt=${ckpt} --patch-size=${patch_size} --sample=${sample3} --epoch=${epoch}
# save ckpt to local.
# sudo
cp checkpoints/t-net-best.pt checkpoints_old/${name}-s-${sample3}-lr-${lr3}.pt
# validate:
CUDA_VISIBLE_DEVICES=${gpu} python validate.py -dg -m=t-net --model=${model} --tolerance_loss=${t} --val_img=${val_img} --val_trimap=${val_tmp} --val_matte=${val_mat} --val_fg=${val_fg} --val_bg=${val_bg} --out=${val_out} --ckpt_path=${ckpt}/t-net-best.pt --patch-size=10000
# save data to local.
# sudo
cp -r data/val/trimap/ data_old/${name}-s-${sample3}-lr-${lr3}/
# save date to /share.
# sudo cp -r data/val/trimap/ /share/data/${name}-s-${sample3}-lr-${lr3}/
# sudo
mv checkpoints/t-net-best.pt checkpoints/${name}-s-${sample3}-lr-${lr3}.pt