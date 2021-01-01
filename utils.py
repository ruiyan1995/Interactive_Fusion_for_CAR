import os
import sys
import json
import pickle
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.ops.roi_align import roi_align


def path_check(path):
    if not os.path.exists(path):
        os.makedirs(path)

def Euclidean_loss(preds, gts):
    # <V, T-obs_len, P, d> d will be 2 or 4 or any dim
    euc_loss = (preds - gts)**2
    euc_loss = euc_loss.sum(-1).sum(-2)
    return euc_loss


def build_region_feas(feature_maps, boxes_list, output_crop_size=[3, 3], img_size=[224, 224]):
    # Building feas for each bounding box by using RoI Align
    # feature_maps:[N,C,H,W], where N=b*T
    IH, IW = img_size
    FH, FW = feature_maps.size()[-2:]  # Feature_H, Feature_W
    region_feas = roi_align(feature_maps, boxes_list, output_crop_size, spatial_scale=float(
        FW)/IW)  # b*T*K, C, S, S; S denotes output_size
    return region_feas.view(region_feas.size(0), -1)  # b*T*K, D*S*S


def to_one_hot(indices, max_index):
    """Get one-hot encoding of index tensors."""
    zeros = torch.zeros(
        indices.size()[0], max_index, dtype=torch.float32, device=indices.device)
    return zeros.scatter_(1, indices.unsqueeze(1), 1)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def box_to_normalized(boxes_tensor, crop_size=[224, 224], mode='list'):
    # tensor to list, and [cx, cy, w, h] --> [x1, y1, x2, y2]
    new_boxes_tensor = boxes_tensor.clone()
    new_boxes_tensor[..., 0] = (
        boxes_tensor[..., 0]-boxes_tensor[..., 2]/2.0)*crop_size[0]
    new_boxes_tensor[..., 1] = (
        boxes_tensor[..., 1]-boxes_tensor[..., 3]/2.0)*crop_size[1]
    new_boxes_tensor[..., 2] = (
        boxes_tensor[..., 0]+boxes_tensor[..., 2]/2.0)*crop_size[0]
    new_boxes_tensor[..., 3] = (
        boxes_tensor[..., 1]+boxes_tensor[..., 3]/2.0)*crop_size[1]
    if mode == 'list':
        boxes_list = []
        for boxes in new_boxes_tensor:
            boxes_list.append(boxes)
        return boxes_list
    elif mode == 'tensor':
        return new_boxes_tensor


def save_pkl(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def get_mask(box_idt):
    # types: HO:0, OO:1, HH:2
    V, T, P = box_idt.size()
    mask = torch.zeros((V, T, P, P))
    for v in range(V):
        for t in range(T):
            for p in range(P):
                for pp in range(P):
                    if pp != p:
                        # pair_data = torch.cat([inputs[v,t,p,:], inputs[v,t,pp,:]], dim=-1).unsqueeze(1).unsqueeze(1).unsqueeze(1) # <1,1,1,2d>
                        if box_idt[v, t, p] == 0 or box_idt[v, t, pp] == 0:  # 'empty-other'
                            mask[v, t, p, pp] = 0
                        elif box_idt[v, t, p] == 1 and box_idt[v, t, pp] == 1:  # 'hand-hand'
                            mask[v, t, p, pp] = 1
                        elif box_idt[v, t, p] == 2 and box_idt[v, t, pp] == 2:  # 'obj-obj'
                            mask[v, t, p, pp] = 4
                        elif (box_idt[v, t, p] + box_idt[v, t, pp]) == 3:  # 'hand-obj'
                            mask[v, t, p, pp] = 2
                        else:
                            print(box_idt[v, t, p], box_idt[v, t, pp])
                            print('error')
#                                 exit(0)
    return mask


def get_fast_mask(box_idt):
    # fast version for getting mask, it takes only 25s over 100,000 video samples.
    # box_idt: 0 is for none, 1 is for hand, and 2 for object.
    # mask: 0: empty-other; 1: hand-hand; 2: hand-obj; 4: obj-obj
    V, T, P = box_idt.size()
    mask = torch.zeros((V, T, P, P))
    for v in range(V):
        for t in range(T):
            a = box_idt[v, t]
            b = box_idt[v, t]
#                 print(a.size(), b.size(), (a*b).size())
            a = a.view(-1, 1)
            b = b.view(1, -1)
#                 print(a.size(), b.size(), (a*b).size())
            mask[v, t] = a*b
            mask[v, t].fill_diagonal_(0)

    if torch.cuda.is_available():
        return mask.cuda()
    else:
        return mask


def get_bin_mask(full_mask, type_ids=None):
    # type_id: 0: empty-other; 1: hand-hand; 2: hand-obj; 4: obj-obj
    zeros = torch.zeros_like(full_mask)
    ones = torch.ones_like(full_mask)
    if len(type_ids) == 1:
        bin_mask = torch.where(
            full_mask == type_ids[0], ones, zeros)  # [V*T*P]
    elif len(type_ids) > 1:
        bin_mask = full_mask
        for type_id in type_ids:
            bin_mask = torch.where(bin_mask == type_id,
                                   ones-2, bin_mask)  # [V*T*P]
        bin_mask = torch.where(bin_mask == -1, ones, zeros)  # [V*T*P]
    else:
        bin_mask = ones
    return bin_mask
