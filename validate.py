import utils
import torch
import random
import logging
import argparse
import numpy as np
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader

from dataloader.dataset import TestDataset, ValDataset
from networks.loss import matting_loss
from networks.matting_model import MattingModel

"""================================================== Arguments ================================================="""
parser = argparse.ArgumentParser('Portrait Matting Testing Arguments.')

parser.add_argument('--val_img',        type=str,   default='',   help='training images.')
parser.add_argument('--val_trimap',     type=str,   default='',   help='intermediate trimaps.')
parser.add_argument('--val_matte',      type=str,   default='',   help='final mattes.')
parser.add_argument('--val_fg',         type=str,   default='',   help='val fg.')
parser.add_argument('--val_bg',         type=str,   default='',   help='val bg.')
parser.add_argument('--out',            type=str,   default='',   help='val image out.')
parser.add_argument('--ckpt_path',      type=str,   default='',   help='checkpoints.')
parser.add_argument('--batch',          type=int,   default=1,    help='input batch size for train')
parser.add_argument('--patch-size',     type=int,   default=480,  help='patch size of input images.')
parser.add_argument('--seed',           type=int,   default=42,   help='random seed.')

parser.add_argument('--model', type=str, choices=['p', 'm', 'g'], default='p', help='p = PSPNet, m = MobileNetV2, g = GFM')

parser.add_argument('--tolerance_loss',       type=bool,   default=False,   help='tolerance loss.')

parser.add_argument('-t', '--random-trimap', action='store_true', help='random generate trimap')
parser.add_argument('-d', '--debug', action='store_true', help='log for debug.')
parser.add_argument('-g', '--gpu',   action='store_true', help='use gpu.')
parser.add_argument('-m', '--mode',  type=str, choices=['end2end', 'f-net', 'm-net', 't-net'], default='end2end', help='working mode.')

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
logger = logging.getLogger('test')
if args.debug:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

logger.debug(args)

"""================================================ Load DataSet ================================================"""
data = ValDataset(args)
data_loader = DataLoader(data, batch_size=args.batch)

"""================================================ Build Model ================================================="""
matting_model = MattingModel(0, not (args.gpu and torch.cuda.is_available()), args.mode, args.model)


matting_model.resume_from_ckpt(torch.load(args.ckpt_path))


"""------------ Test --------------"""
matting_model.eval()
val_loss = 0
with torch.no_grad():
    for idx, batch in enumerate(data_loader):
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
        logger.debug(f'Batch: {idx + 1}/{len(data_loader)} \t' 
                     f'Validation T-Loss: {t_loss.item() if t_loss is not None else 0:8.5f} \t'
                     f'M-Loss: {m_loss.item() if m_loss is not None else 0:8.5f}')

        utils.save_images(args.out, batch['name'], ptp, pmu, gt_trimap_3, logger)

average_loss = val_loss / len(data_loader)
matting_model.losses.append(average_loss)

""" Write Log and Tensorboard """
logger.info(f'Loss:{average_loss}')
