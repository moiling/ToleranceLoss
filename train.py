from tensorboardX import SummaryWriter

import torch
import random
import logging
import argparse
import numpy as np
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader

import utils
from dataloader.dataset import TrainDataset, ValDataset
from dataloader.prefetcher import PreFetcher
from networks.loss import matting_loss
from networks.matting_model import MattingModel

"""================================================== Arguments ================================================="""
parser = argparse.ArgumentParser('Portrait Matting Training Arguments.')

parser.add_argument('--img',        type=str,   default='D:/Dataset/ZT/zt4k/train/image',   help='training images.')
parser.add_argument('--trimap',     type=str,   default='D:/Dataset/ZT/zt4k/train/trimap',   help='intermediate trimaps.')
parser.add_argument('--matte',      type=str,   default='D:/Dataset/ZT/zt4k/train/alpha',   help='final mattes.')
parser.add_argument('--fg',         type=str,   default='D:/Dataset/ZT/zt4k/train/fg',   help='fg for loss.')
parser.add_argument('--bg',         type=str,   default='D:/Dataset/ZT/zt4k/train/bg',   help='bg for loss.')
parser.add_argument('--val-out',    type=str,   default='data/val',   help='val image out.')
parser.add_argument('--val-img',    type=str,   default='D:/Dataset/ZT/zt4k/test_small/image',   help='val images.')
parser.add_argument('--val-trimap', type=str,   default='D:/Dataset/ZT/zt4k/test_small/trimap',   help='intermediate val trimaps.')
parser.add_argument('--val-matte',  type=str,   default='D:/Dataset/ZT/zt4k/test_small/alpha',   help='val mattes.')
parser.add_argument('--val-fg',     type=str,   default='D:/Dataset/ZT/zt4k/test_small/fg',   help='val fg.')
parser.add_argument('--val-bg',     type=str,   default='D:/Dataset/ZT/zt4k/test_small/bg',   help='val bg.')
parser.add_argument('--ckpt',       type=str,   default='checkpoints',   help='checkpoints.')
parser.add_argument('--batch',      type=int,   default=2,    help='input batch size for train')
parser.add_argument('--val-batch',  type=int,   default=1,    help='input batch size for val')
parser.add_argument('--epoch',      type=int,   default=10,   help='number of epochs.')
parser.add_argument('--sample',     type=int,   default=10, help='number of samples. -1 means all samples.')
parser.add_argument('--lr',         type=float, default=1e-5, help='learning rate while training.')
parser.add_argument('--patch-size', type=int,   default=480,  help='patch size of input images.')
parser.add_argument('--seed',       type=int,   default=42,   help='random seed.')

parser.add_argument('--model', type=str, choices=['p', 'm', 'g', 'a'], default='p', help='p = PSPNet, m = MobileNetV2, g = GFM')

parser.add_argument('--tolerance_loss',       type=bool,   default=False,   help='tolerance loss.')

parser.add_argument('-t', '--random-trimap', action='store_true', help='random generate trimap')
parser.add_argument('-d', '--debug',         action='store_true', help='log for debug.')
parser.add_argument('-g', '--gpu',           action='store_true', help='use gpu.')
parser.add_argument('-r', '--resume',        action='store_true', help='load a previous checkpoint if exists.')
parser.add_argument('--hr',                  action='store_true', help='lr or hr.')

parser.add_argument('-m', '--mode', type=str, choices=['end2end', 'f-net', 'm-net', 't-net'], default='t-net', help='working mode.')

args = parser.parse_args()

"""================================================= Presetting ================================================="""
torch.set_flush_denormal(True)  # flush cpu subnormal float.
cudnn.enabled = True
cudnn.benchmark = True
# random seed.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
# logger
logging.basicConfig(level=logging.INFO, format='[%(asctime)-15s] [%(name)s:%(lineno)s] %(message)s')
logger = logging.getLogger('train')
tb_logger = SummaryWriter()
if args.debug:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

logger.debug(args)

"""================================================ Load DataSet ================================================"""
# train
train_data = TrainDataset(args)
train_data_loader = DataLoader(train_data, batch_size=args.batch, drop_last=True, shuffle=True)
train_data_loader = PreFetcher(train_data_loader)
# val
val_data = ValDataset(args)
val_data_loader = DataLoader(val_data, batch_size=args.val_batch)

"""================================================ Build Model ================================================="""
matting_model = MattingModel(args.lr, not (args.gpu and torch.cuda.is_available()), args.mode, args.model)

if args.resume:
    matting_model.resume(args.ckpt)

"""================================================= Main Loop =================================================="""
for epoch in range(matting_model.start_epoch, args.epoch + 1):
    """--------------- Train ----------------"""
    matting_model.train()
    logger.info(f'Epoch: {epoch}/{args.epoch}')

    for idx, batch in enumerate(train_data_loader):
        """ Load Batch Data """
        img         = batch['img']
        trimap_3    = batch['trimap_3']
        gt_trimap_3 = batch['trimap_3']
        gt_matte    = batch['matte']
        gt_fg       = batch['fg'] if 'fg' in batch.keys() else None
        gt_bg       = batch['bg'] if 'bg' in batch.keys() else None

        if args.gpu and torch.cuda.is_available():
            img         = img.cuda()
            trimap_3    = trimap_3.cuda()
            gt_trimap_3 = gt_trimap_3.cuda()
            gt_matte    = gt_matte.cuda()
            if gt_fg is not None and gt_bg is not None:
                gt_fg   = gt_fg.cuda()
                gt_bg   = gt_bg.cuda()
        else:
            img         = img.cpu()
            trimap_3    = trimap_3.cpu()
            gt_trimap_3 = gt_trimap_3.cpu()
            gt_matte    = gt_matte.cpu()
            if gt_fg is not None and gt_bg is not None:
                gt_fg   = gt_fg.cpu()
                gt_bg   = gt_bg.cpu()

        """ Forward """
        ptp, pmu = matting_model.forward(img, trimap_3)

        """ Backward """
        t_loss, m_loss = matting_model.backward(img, ptp, pmu, gt_trimap_3, gt_matte, gt_fg, gt_bg, tolerance_loss=args.tolerance_loss)

        """ Write Log and Tensorboard """
        logger.debug(f'{args.mode}\t Batch: {idx + 1}/{len(train_data_loader.orig_loader)} \t'
                     f'T-Loss: {t_loss.item() if t_loss is not None else 0:8.5f} \t'
                     f'M-Loss: {m_loss.item() if m_loss is not None else 0:8.5f}')

        step = (epoch - 1) * len(train_data_loader.orig_loader) + idx
        if step % 100 == 0:
            if t_loss is not None:
                tb_logger.add_scalar('TRAIN/T-Loss', t_loss.item(), step)
            if m_loss is not None:
                tb_logger.add_scalar('TRAIN/M-Loss', m_loss.item(), step)

    """------------ Validation --------------"""
    matting_model.eval()
    val_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(val_data_loader):
            """ Load Batch Data """
            img         = batch['img']
            trimap_3    = batch['trimap_3']
            gt_trimap_3 = batch['trimap_3']
            gt_matte    = batch['matte']
            gt_fg       = batch['fg'] if 'fg' in batch.keys() else None
            gt_bg       = batch['bg'] if 'bg' in batch.keys() else None

            if args.gpu and torch.cuda.is_available():
                img         = img.cuda()
                trimap_3    = trimap_3.cuda()
                gt_trimap_3 = gt_trimap_3.cuda()
                gt_matte    = gt_matte.cuda()
                if gt_fg is not None and gt_bg is not None:
                    gt_fg   = gt_fg.cuda()
                    gt_bg   = gt_bg.cuda()
            else:
                img         = img.cpu()
                trimap_3    = trimap_3.cpu()
                gt_trimap_3 = gt_trimap_3.cpu()
                gt_matte    = gt_matte.cpu()
                if gt_fg is not None and gt_bg is not None:
                    gt_fg   = gt_fg.cpu()
                    gt_bg   = gt_bg.cpu()

            """ Forward """
            ptp, pmu = matting_model.forward(img, trimap_3)

            """ Calculate Loss """
            t_loss, m_loss = matting_loss(img, ptp, pmu, gt_trimap_3, gt_matte, args.mode, gt_fg, gt_bg, tolerance_loss=args.tolerance_loss)

            if args.mode == 't-net':
                val_loss += t_loss.item()
            if args.mode == 'm-net':
                val_loss += m_loss.item()
            if args.mode == 'end2end':
                val_loss += t_loss.item() + m_loss.item()

            """ Write Log and Save Images """
            logger.debug(f'Batch: {idx + 1}/{len(val_data_loader)} \t' 
                         f'Validation T-Loss: {t_loss.item() if t_loss is not None else 0:8.5f} \t'
                         f'M-Loss: {m_loss.item() if m_loss is not None else 0:8.5f}')

            utils.save_images(args.val_out, batch['name'], ptp, pmu, gt_trimap_3, logger)

    average_loss = val_loss / len(val_data_loader)
    matting_model.losses.append(average_loss)

    """ Write Log and Tensorboard """
    tb_logger.add_scalar('TEST/Loss', average_loss, epoch)
    logger.info(f'Loss:{average_loss}')

    """------------ Save Model --------------"""
    if min(matting_model.losses) == average_loss:
        logger.info('Minimal loss so far.')
        matting_model.save(args.ckpt, epoch, best=True)
    else:
        matting_model.save(args.ckpt, epoch, best=False)
