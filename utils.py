import os
import sys
import json
import pickle
import argparse
import torch
import shutil
import matplotlib.pyplot as plt
import numpy as np
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
    zeros = torch.zeros(indices.size()[0], max_index, dtype=torch.float32, device=indices.device)
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

def load_args():
    parser = argparse.ArgumentParser(description='Smth-Smth example training')
    parser.add_argument('--config', '-c', help='json config file path')
    parser.add_argument('--eval_only', '-e', action='store_true',
                        help="evaluate trained model on validation data.")
    parser.add_argument('--resume', '-r', action='store_true',
                        help="resume training from a given checkpoint.")
    parser.add_argument('--gpus', '-g', help="GPU ids to use. Please enter a comma separated list")
    parser.add_argument('--use_cuda', action='store_true', help="to use GPUs")
    args = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    return args


def remove_module_from_checkpoint_state_dict(state_dict):
    """
    Removes the prefix `module` from weight names that gets added by
    torch.nn.DataParallel()
    """
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def load_json_config(path):
    """ loads a json config file"""
    with open(path) as data_file:
        config = json.load(data_file)
        config = config_init(config)
    return config


def config_init(config):
    """ Some of the variables that should exist and contain default values """
    if "augmentation_mappings_json" not in config:
        config["augmentation_mappings_json"] = None
    if "augmentation_types_todo" not in config:
        config["augmentation_types_todo"] = None
    return config


def setup_cuda_devices(args):
    device_ids = []
    # Take the first GPU from the GPU args
    device = torch.device("cuda:{gpu}".format(gpu=args.gpus[:1]) if args.use_cuda else "cpu")
    if device.type == "cuda":
        device_ids = [int(i) for i in args.gpus.split(',')]
    return device, device_ids


def save_checkpoint(state, is_best, config, filename='checkpoint.pth.tar'):
    checkpoint_path = os.path.join(config['output_dir'], config['model_name'], filename)
    model_path = os.path.join(config['output_dir'], config['model_name'], 'model_best.pth.tar')
    torch.save(state, checkpoint_path)

    if state['epoch'] % config.get('saving_checkpoints_iter', 5) == 0:
        print(" > Save model found at this epoch because iteration module {iter}. Saving ..."
              .format(iter=config.get('saving_checkpoints_iter', 5)))
        iter_model_path = os.path.join(config['output_dir'],
                                       config['model_name'],
                                       'model_{}.pth.tar'.format(state['epoch']))
        shutil.copyfile(checkpoint_path, iter_model_path)

    if is_best:
        print(" > Best model found at this epoch. Saving ...")
        shutil.copyfile(checkpoint_path, model_path)


def save_results(logits_matrix, targets_list,
                 class_to_idx, args):
    """
    Saves the predicted logits matrix, true labels, sample ids and class
    dictionary for further analysis of results
    """
    print("Saving inference results ...")
    path_to_save = os.path.join(
        args.ckpt, args.logname + '_' "test_results.pkl")

    with open(path_to_save, "wb") as f:
        pickle.dump([logits_matrix, targets_list,
                     class_to_idx], f)


def save_images_for_debug(dir_img, imgs):
    """
    2x3x12x224x224 --> [BS, C, seq_len, H, W]
    """
    print("Saving images to {}".format(dir_img))
    from matplotlib import pylab as plt
    imgs = imgs.permute(0, 2, 3, 4, 1)  # [BS, seq_len, H, W, C]
    imgs = imgs.mul(255).numpy()
    if not os.path.exists(dir_img):
        os.makedirs(dir_img)
    print(imgs.shape)
    for batch_id, batch in enumerate(imgs):
        batch_dir = os.path.join(dir_img, "batch{}".format(batch_id + 1))
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)
        for j, img in enumerate(batch):
            plt.imsave(os.path.join(batch_dir, "frame{%04d}.png" % (j + 1)),
                       img.astype("uint8"))



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def box_to_normalized(boxes_tensor, crop_size=[224,224], mode='list'):
    # tensor to list, and [cx, cy, w, h] --> [x1, y1, x2, y2]
    new_boxes_tensor = boxes_tensor.clone()
    new_boxes_tensor[...,0] = (boxes_tensor[...,0]-boxes_tensor[...,2]/2.0)*crop_size[0]
    new_boxes_tensor[...,1] = (boxes_tensor[...,1]-boxes_tensor[...,3]/2.0)*crop_size[1]
    new_boxes_tensor[...,2] = (boxes_tensor[...,0]+boxes_tensor[...,2]/2.0)*crop_size[0]
    new_boxes_tensor[...,3] = (boxes_tensor[...,1]+boxes_tensor[...,3]/2.0)*crop_size[1]
    if mode == 'list':
        boxes_list = []
        for boxes in new_boxes_tensor:
            boxes_list.append(boxes)
        return boxes_list
    elif mode == 'tensor':
        return new_boxes_tensor

"""
def convert_box(boxes_list, crop_size=[224,224]):
    # [cx, cy, w, h] --> [x1, y1, x2, y2]
    new_boxes_list = []
    for boxes in boxes_list:
        new_boxes = []
        for box in boxes:
            cx, cy, w, h = box
            cx, w = cx*crop_size[0], w*crop_size[0]
            cy, h = cy*crop_size[1], h*crop_size[1]
            new_boxes.append([cx-w/2.0, cy-h/2.0, cx+w/2.0, cy+h/2.0])
        new_boxes_list.append(torch.tensor(new_boxes).cuda())
    return new_boxes_list

def tensor_to_list(boxes_tensor):
    boxes_list = []
    for boxes in boxes_tensor:
        boxes_list.append(boxes)
    return boxes_list

"""



def results_collect(sample_meta, result, sampled=True, last_half=True):
    tmp_dict = {}
    for i, meta in enumerate(sample_meta):
        video_id, frame_ids = int(meta[0]), list(map(int, meta[1:]))
        # print(video_id, frame_ids)
        if sampled:
            frame_ids = frame_ids[::2]
        if last_half:
            middle = len(frame_ids)//2
            frame_ids = frame_ids[middle:]
        else:
            frame_ids = frame_ids[-1]
        re = {'pred_pos':result['pred_pos'][i], 'gt_pos':result['gt_pos'][i], 'pred_offset':result['pred_offset'][i], 'gt_offset':result['gt_offset'][i]}
        tmp_dict[video_id] = {'ids': frame_ids, 'results': re}
        # print(frame_ids, re)
        #print(video_id, tmp_dict[video_id])
    return tmp_dict

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
                        if pp!=p:
                            # pair_data = torch.cat([inputs[v,t,p,:], inputs[v,t,pp,:]], dim=-1).unsqueeze(1).unsqueeze(1).unsqueeze(1) # <1,1,1,2d>
                            if box_idt[v, t, p]==0 or box_idt[v, t, pp]==0: #'empty-other'
                                mask[v, t, p, pp] = 0
                            elif box_idt[v, t, p]==1 and box_idt[v, t, pp]==1: #'hand-hand'
                                mask[v, t, p, pp] = 1
                            elif box_idt[v, t, p]==2 and box_idt[v, t, pp]==2: #'obj-obj'
                                mask[v, t, p, pp] = 4
                            elif (box_idt[v, t, p] + box_idt[v, t, pp])==3: #'hand-obj'
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
                a = box_idt[v,t]
                b = box_idt[v,t]
#                 print(a.size(), b.size(), (a*b).size())
                a = a.view(-1,1)
                b = b.view(1,-1)
#                 print(a.size(), b.size(), (a*b).size())
                mask[v,t] = a*b
                mask[v,t].fill_diagonal_(0)
    
    if torch.cuda.is_available():
        return mask.cuda()
    else:
        return mask
'''
def get_bin_mask(full_mask, type_id=None):
    # type_id: 0: empty-other; 1: hand-hand; 2: hand-obj; 4: obj-obj
    zeros = torch.zeros_like(full_mask)
    ones = torch.ones_like(full_mask)
    if type(type_id) == int:
        bin_mask = torch.where(full_mask==type_id, ones, zeros) # [V*T*P]
    else:
        bin_mask = ones
    return bin_mask
'''
def get_bin_mask(full_mask, type_ids=None):
    # type_id: 0: empty-other; 1: hand-hand; 2: hand-obj; 4: obj-obj
    zeros = torch.zeros_like(full_mask)
    ones = torch.ones_like(full_mask)
    if len(type_ids) == 1:
        bin_mask = torch.where(full_mask==type_ids[0], ones, zeros) # [V*T*P]
    elif len(type_ids) >1:
        bin_mask = full_mask
        for type_id in type_ids:
            bin_mask = torch.where(bin_mask==type_id, ones-2, bin_mask) # [V*T*P]
        bin_mask = torch.where(bin_mask==-1, ones, zeros) # [V*T*P]
    else:
        bin_mask = ones
    return bin_mask


def mAP(submission_array, gt_array):
    """ Returns mAP, weighted mAP, and AP array """
    # print(submission_array[0], gt_array[0])
    # print(submission_array.shape, gt_array.shape)
    
    
    m_aps = []
    n_classes = submission_array.shape[1]
    for oc_i in range(n_classes):
        sorted_idxs = np.argsort(-submission_array[:, oc_i])
        # print(sorted_idxs,gt_array[:, oc_i][sorted_idxs])
        # print(len(sorted_idxs))
        # exit(0)
        tp = gt_array[:, oc_i][sorted_idxs] == 1
        fp = np.invert(tp)
        n_pos = tp.sum()
        # print('n_pos:',n_pos)
        if n_pos < 0.1:
            m_aps.append(float('nan'))
            continue
        fp.sum()
        # print('fp:', fp)
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs+t_pcs).astype(float)
        # print('prec:', prec)
        avg_prec = 0
        for i in range(submission_array.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.array(m_aps)
    m_ap = np.nanmean(m_aps)
    w_ap = (m_aps * gt_array.sum(axis=0) / gt_array.sum().sum().astype(float))
    return m_ap*100, w_ap*100, m_aps*100


def charades_map(submission_array, gt_array):
    """ 
    Approximate version of the charades evaluation function
    For precise numbers, use the submission file with the official matlab script
    """
    fix = submission_array.copy()
    empty = np.sum(gt_array, axis=1)==0
    fix[empty, :] = np.NINF
    return mAP(fix, gt_array)
