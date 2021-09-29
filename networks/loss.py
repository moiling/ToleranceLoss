import torch
import torch.nn as nn
import torch.nn.functional as F


def comp_loss(img, pred_matte, gt_fg=None, gt_bg=None, mask=None):
    merge = gt_fg * pred_matte + gt_bg * (1 - pred_matte)
    if mask is None:
        return F.l1_loss(merge, img)
    else:
        return F.l1_loss(merge * mask, img * mask, reduction='sum') / (torch.sum(mask) + 1e-8)


def alpha_loss(pred_matte, gt_matte, mask=None):
    if mask is None:
        return F.l1_loss(gt_matte, pred_matte)
    else:
        return F.l1_loss(gt_matte * mask, pred_matte * mask, reduction='sum') / (torch.sum(mask) + 1e-8)


def class_loss(pred_trimap_prob, gt_trimap_3):
    gt_trimap_type = gt_trimap_3.argmax(dim=1)   # [B, H, W]
    return __class_loss_type(pred_trimap_prob, gt_trimap_type)


def __class_loss_type(pred_trimap_prob, gt_trimap_type):
    criterion = nn.CrossEntropyLoss()
    return criterion(pred_trimap_prob, gt_trimap_type)


def tolerance_class_loss(pred_trimap_prob, gt_trimap_3):
    gt_type = torch.zeros_like(gt_trimap_3.argmax(dim=1))  # [B, H, W], all target type = 0
    # CrossEntropyLoss = softmax + log + NLLLoss
    # soft max
    softmax_func = nn.Softmax(dim=1)
    pred_trimap_softmax = softmax_func(pred_trimap_prob)
    # BUF
    b = pred_trimap_softmax[:, 0:1, ...]
    u = pred_trimap_softmax[:, 1:2, ...]
    f = pred_trimap_softmax[:, 2:3, ...]

    gt_argmax = gt_trimap_3.argmax(dim=1).unsqueeze(dim=1)
    b_mask = (gt_argmax == 0)
    u_mask = (gt_argmax == 1)
    f_mask = (gt_argmax == 2)

    # B => true = b+u, false = f
    # U => true = u, false = f+b
    # F => true = f+u, false = b
    pred_true_softmax = b_mask * (b + u) + u_mask * u + f_mask * (f + u)
    pred_false_softmax = b_mask * f + u_mask * (f + b) + f_mask * b

    pred_softmax = torch.cat((pred_true_softmax, pred_false_softmax), dim=1)  # [B, 2, H, W]

    # log + NLLLoss
    nll_loss_func = nn.NLLLoss()
    cross_entropy_loss = nll_loss_func(torch.log(pred_softmax), gt_type)
    return cross_entropy_loss


def matting_loss(img, pred_trimap_prob, pred_matte_u, gt_trimap_3, gt_matte, mode, gt_fg=None, gt_bg=None, tolerance_loss=False):
    mask = gt_trimap_3[:, 1:2, ...]
    mask = mask.detach()

    t_loss, m_loss = None, None
    if mode == 't-net':
        t_loss = class_loss(pred_trimap_prob, gt_trimap_3)
        if tolerance_loss:
            t_loss += 10 * tolerance_class_loss(pred_trimap_prob, gt_trimap_3)
        return t_loss, m_loss
    if mode == 'm-net':
        if gt_fg is not None and gt_bg is not None:
            m_loss = (0.5 * alpha_loss(pred_matte_u, gt_matte, mask) +
                      0.5 * comp_loss(img, pred_matte_u, gt_fg, gt_bg, mask))
        else:
            m_loss = alpha_loss(pred_matte_u, gt_matte, mask)
        return t_loss, m_loss

    mask = (pred_trimap_prob.softmax(dim=1).argmax(dim=1) == 1).float().unsqueeze(dim=1)
    mask = mask.detach()
    # end2end
    t_loss = class_loss(pred_trimap_prob, gt_trimap_3)
    if tolerance_loss:
        t_loss += 10 * tolerance_class_loss(pred_trimap_prob, gt_trimap_3)

    if gt_fg is not None and gt_bg is not None:
        m_loss = (0.5 * alpha_loss(pred_matte_u, gt_matte, mask) +
                  0.5 * comp_loss(img, pred_matte_u, gt_fg, gt_bg, mask))
    else:
        m_loss = alpha_loss(pred_matte_u, gt_matte, mask)
    return t_loss, m_loss


def correction_loss(pred_trimap_prob, pred_matte):
    pred_matte = pred_matte.detach()
    pred_trimap_type = pred_trimap_prob.detach().softmax(dim=1).argmax(dim=1)    # b=0 u=1 f=2
    correction_region = (pred_matte != 0 & pred_trimap_type == 0) | (pred_matte != 1 & pred_trimap_type == 2)
    target_trimap_type = pred_trimap_type
    target_trimap_type[correction_region] = 1
    return __class_loss_type(pred_trimap_prob, target_trimap_type)
