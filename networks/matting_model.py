import logging
import os
import time

import torch

from networks.loss import matting_loss
from networks.mnet.dimnet import DIMNet
from networks.tnet.AimNet import AimNet
from networks.tnet.gfm import GFM
from networks.tnet.mobilenet_wrapper import MobileNetWrapper
from networks.tnet.pspnet import PSPNet


class MattingModel:

    cpu = False
    mode = ''

    t_optimizer = None
    m_optimizer = None
    t_net = None
    m_net = None

    start_epoch = 1
    losses = []

    logger = logging.getLogger('MattingModel')
    logger.setLevel(logging.DEBUG)

    def __init__(self, lr, cpu=False, mode='t-net', model='p'):
        self.cpu = cpu
        self.mode = mode

        # Build Model
        if model == 'p':
            self.t_net = PSPNet(pretrain=True)
        elif model == 'm':
            self.t_net = MobileNetWrapper(pretrain=False)
        elif model == 'g':
            self.t_net = GFM(pretrain=True)
        else:
            self.t_net = AimNet(pretrained=True)

        self.m_net = DIMNet()

        self.t_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.t_net.parameters()), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
        self.m_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.m_net.parameters()), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)

        if not cpu:
            self.t_net.cuda()
            self.m_net.cuda()
        else:
            self.t_net.cpu()
            self.m_net.cpu()

    def forward(self, img, trimap_3=None):
        if self.mode == 't-net':
            pred_trimap_prob = self.t_net(img)  # [B, C(BUF=3),     H, W]
            return pred_trimap_prob, None
        if self.mode == 'm-net':
            concat = torch.cat([img, trimap_3], dim=1)
            pred_matte_u = self.m_net(concat)
            return None, pred_matte_u
        if self.mode == 'end2end':
            pred_trimap_prob = self.t_net(img)
            pred_trimap_softmax = pred_trimap_prob.softmax(dim=1)
            concat = torch.cat([img, pred_trimap_softmax], dim=1)
            pred_matte_u = self.m_net(concat)
            return pred_trimap_prob, pred_matte_u

    def backward(self, img, pred_trimap_prob, pred_matte_u, gt_trimap_3, gt_matte, gt_fg=None, gt_bg=None, tolerance_loss=False):
        """ Calculate Loss """
        t_loss, m_loss = matting_loss(img, pred_trimap_prob, pred_matte_u, gt_trimap_3, gt_matte, self.mode, gt_fg, gt_bg, tolerance_loss=tolerance_loss)

        """ Back Propagate """
        """ Update Parameters """
        if self.mode == 't-net':
            self.t_optimizer.zero_grad()
            t_loss.backward()
            self.t_optimizer.step()
            return t_loss, m_loss

        if self.mode == 'm-net':
            self.m_optimizer.zero_grad()
            m_loss.backward()
            self.m_optimizer.step()
            return t_loss, m_loss

        if self.mode == 'end2end':
            self.t_optimizer.zero_grad()
            self.m_optimizer.zero_grad()
            t_loss.backward(retain_graph=True)
            m_loss.backward()
            self.t_optimizer.step()
            self.m_optimizer.step()
            return t_loss, m_loss

    def train(self):
        if self.mode == 't-net':
            self.t_net.train()
            self.m_net.eval()
            return

        if self.mode == 'm-net':
            self.t_net.eval()
            self.m_net.train()
            return

        if self.mode == 'end2end':
            self.t_net.train()
            self.m_net.train()

    def eval(self):
        self.t_net.eval()
        self.m_net.eval()

    def resume_from_ckpt(self, ckpt):
        if ckpt is None:
            self.start_epoch = 1
            self.losses = [1e5]
            return

        if ckpt['t_net'] and self.t_net:
            self.t_net.load_state_dict(ckpt['t_net'])
        if ckpt['m_net'] and self.m_net:
            self.m_net.load_state_dict(ckpt['m_net'])

        if ckpt['t_optimizer'] and self.t_optimizer:
            self.t_optimizer.load_state_dict(ckpt['t_optimizer'])
        if ckpt['m_optimizer'] and self.m_optimizer:
            self.m_optimizer.load_state_dict(ckpt['m_optimizer'])

        self.start_epoch = ckpt['epoch'] + 1
        self.losses = ckpt['losses']
        self.logger.debug(f'Load Checkpoint => losses: {self.losses}, epoch: {self.start_epoch - 1}')

    def resume(self, checkpoint_dir):
        ckpt = self._load_checkpoint(checkpoint_dir)
        self.resume_from_ckpt(ckpt)

    def save(self, checkpoint_dir, epoch, best=False):

        if best:
            file_name = f'{self.mode}-best.pt'
        else:
            file_name = f'{self.mode}-last.pt'

        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, file_name)

        torch.save({
            'epoch': epoch,
            'losses': self.losses,
            't_net': self.t_net.state_dict(),
            'm_net': self.m_net.state_dict(),
            't_optimizer': self.t_optimizer.state_dict(),
            'm_optimizer': self.m_optimizer.state_dict()
        }, path)

        self.logger.debug(f'Saving Model to "{path}"')

    def _load_checkpoint(self, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            return None

        best_model = None
        last_models, best_models = [], []

        for name in os.listdir(checkpoint_dir):
            if 'best' in name:
                best_models.append(name)
            else:
                last_models.append(name)

        # get last model.
        if best_models:
            best_models.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
            best_model = best_models[-1]
        elif last_models:
            last_models.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
            best_model = last_models[-1]

        if best_model:
            path = os.path.join(checkpoint_dir, best_model)
            self.logger.debug(f'Loading Model from "{path}"')
            checkpoint = torch.load(path)

            # different mode => initial losses & epoch & optimizer_state_dict.
            if not best_model.startswith(self.mode):
                checkpoint['losses'] = [1e5]
                checkpoint['epoch'] = 0
                checkpoint['t_optimizer'] = None
                checkpoint['m_optimizer'] = None
            return checkpoint

        return None
