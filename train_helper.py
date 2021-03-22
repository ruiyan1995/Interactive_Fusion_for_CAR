import torch
import numpy as np
import shutil
import os
import pickle

def create_model(model_name: str, args):
    # create model
    if model_name == 'coord':
        from model.original_model_lib import VideoModelCoord as VideoModel
    elif model_name == 'coord_latent':
        from model.original_model_lib import VideoModelCoordLatent as VideoModel
    elif model_name == 'coord_latent_nl':
        from model.original_model_lib import VideoModelCoordLatentNL as VideoModel
    elif model_name == 'global_coord_latent':
        from model.original_model_lib import VideoModelGlobalCoordLatent as VideoModel
    elif model_name == 'global_coord_latent_nl':
        from model.original_model_lib import VideoModelGlobalCoordLatentNL as VideoModel
    elif model_name == 'global':
        from model.original_model_lib import VideoGlobalModel as VideoModel
    elif model_name == 'global_coord':
        from model.original_model_lib import VideoModelGlobalCoord as VideoModel
    elif model_name == 'region':
        from model.model_lib import VideoRegionModel as VideoModel
    return VideoModel(args)


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest.pth.tar',
                        filename + '_best.pth.tar')


def adjust_learning_rate(args, optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mAP(submission_array, gt_array):
    """ Returns mAP, weighted mAP, and AP array """
    m_aps = []
    n_classes = submission_array.shape[1]
    for oc_i in range(n_classes):
        sorted_idxs = np.argsort(-submission_array[:, oc_i])
        tp = gt_array[:, oc_i][sorted_idxs] == 1
        fp = np.invert(tp)
        n_pos = tp.sum()
        if n_pos < 0.1:
            m_aps.append(float('nan'))
            continue
        fp.sum()
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs+t_pcs).astype(float)
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
    empty = np.sum(gt_array, axis=1) == 0
    fix[empty, :] = np.NINF
    return mAP(fix, gt_array)


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
