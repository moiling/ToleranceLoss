@echo off

::set val_img=D:/Dataset/ZT/zt4k/test/image
::set val_tmp=D:/Dataset/ZT/zt4k/test/perfect_trimap
::set val_mat=D:/Dataset/ZT/zt4k/test/alpha
::set val_fg=D:/Dataset/ZT/zt4k/test/fg
::set val_bg=D:/Dataset/ZT/zt4k/test/bg

set val_img=D:/Dataset/Matting/AIM-500/original_png
set val_tmp=D:/Dataset/Matting/AIM-500/perfect_trimap
set val_mat=D:/Dataset/Matting/AIM-500/mask
set val_fg=
set val_bg=

set out=data
set ckpt_path=checkpoints/base_perfect.pt
set patch_size=1024
set t=True

set model=g

python validate.py -dg -m=t-net --model=%model% --tolerance_loss=%t% --val_img=%val_img% --val_trimap=%val_tmp% --val_matte=%val_mat% --val_fg=%val_fg% --val_bg=%val_bg% --out=%out% --ckpt_path=%ckpt_path% --patch-size=%patch_size%
Pause