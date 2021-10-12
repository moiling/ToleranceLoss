@echo off

::set img=D:/Dataset/ZT/zt4k/train/image
::set tmp=D:/Dataset/ZT/zt4k/train/trimap
::set mat=D:/Dataset/ZT/zt4k/train/alpha
::set fg=D:/Dataset/ZT/zt4k/train/fg
::set bg=D:/Dataset/ZT/zt4k/train/bg
::set val_img=D:/Dataset/ZT/zt4k/test/image
::set val_tmp=D:/Dataset/ZT/zt4k/test/trimap
::set val_mat=D:/Dataset/ZT/zt4k/test/alpha
::set val_fg=D:/Dataset/ZT/zt4k/test/fg
::set val_bg=D:/Dataset/ZT/zt4k/test/bg
set img=D:/Dataset/Matting/AIM-Train/image
set tmp=D:/Dataset/Matting/AIM-Train/perfect_trimap
set mat=D:/Dataset/Matting/AIM-Train/alpha
set fg=D:/Dataset/Matting/AIM-Train/fg
set bg=D:/Dataset/Matting/AIM-Train/bg
set val_img=D:/Dataset/Matting/AIM-500/original_png
set val_tmp=D:/Dataset/Matting/AIM-500/perfect_trimap
set val_mat=D:/Dataset/Matting/AIM-500/mask
set val_fg=
set val_bg=
set val_out=data/val
set ckpt=checkpoints
set patch_size=320
set sample=2000
set epoch=100

set batch=4

set t=True

set model=a

set lr=1e-6

python train.py -dgr -m=t-net --lr=%lr% --model=%model% --batch=%batch%  --tolerance_loss=%t% --img=%img% --trimap=%tmp% --matte=%mat% --fg=%fg% --bg=%bg% --val-fg=%val_fg% --val-bg=%val_bg% --val-out=%val_out% --val-img=%val_img% --val-trimap=%val_tmp% --val-matte=%val_mat% --ckpt=%ckpt% --patch-size=%patch_size% --sample=%sample% --epoch=%epoch%
