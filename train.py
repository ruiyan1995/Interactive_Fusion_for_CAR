import os
import sys
import shutil
import time
import numpy as np
import json
import pickle
import torch
import torch.backends.cudnn as cudnn
from collections import OrderedDict
from callbacks import AverageMeter, Logger
from data_utils.data_loader_frames import VideoFolder
from configs import cfg_init
import train_helper
from utils import str2bool
from train_helper import create_model, save_checkpoint, adjust_learning_rate, accuracy, charades_map, save_results

best_loss = 1000000
best_acc_top1 = 0
best_mAP = 0
args = cfg_init.main_args()


def main():
    global args, best_loss, best_acc_top1, best_mAP
    print(args)

    # create model
    model = create_model(args.model, args)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(
            args.resume), "No checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        if torch.cuda.is_available():
            checkpoint = torch.load(args.resume)
        else:
            checkpoint = torch.load(
                args.resume, map_location=torch.device('cpu'))

        if args.start_epoch is None:
            args.start_epoch = checkpoint['epoch']
        # best_loss = checkpoint['best_loss']
        if args.dataset_name == 'charades':
            best_mAP = checkpoint['best_mAP']
        else:
            best_acc_top1 = checkpoint['best_acc_top1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))

    if args.start_epoch is None:
        args.start_epoch = 0

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()  # device_ids=[0,1,2,3]
    else:
        model = torch.nn.DataParallel(model)
    print(model)
    cudnn.benchmark = True


    # loading box annos
    print('... Loading box annotations might take a minute ...')
    since = time.time()
    with open(args.tracked_boxes, 'rb') as f:
        box_annotations = pickle.load(f)

    print('load box anno takes ', time.time()-since)
    # create training and validation dataset
    if not args.evaluate:
        dataset_train = VideoFolder(root=args.root_frames,
                                    num_boxes=args.num_boxes,
                                    file_input=args.json_data_train,
                                    file_labels=args.json_file_labels,
                                    frames_duration=args.num_frames,
                                    args=args,
                                    is_val=False,
                                    if_augment=args.if_augment,  # modified by Mr. Yan
                                    model=args.model,
                                    anno=box_annotations)
        # create training loader
        print('create training loader')
        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=False) # pin_memory=True, modified by Mr. Yan

    dataset_val = VideoFolder(root=args.root_frames,
                              num_boxes=args.num_boxes,
                              file_input=args.json_data_val,
                              file_labels=args.json_file_labels,
                              frames_duration=args.num_frames,
                              args=args,
                              is_val=True,
                              if_augment=args.if_augment,  # modified by Mr. Yan
                              model=args.model,
                              anno=box_annotations)
    # create validation loader
    print('create validation loader')
    val_loader = torch.utils.data.DataLoader(
        dataset_val, drop_last=False,  # drop_last=True, modified by Mr. Yan
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False # pin_memory=True, modified by Mr. Yan
    )

    optimizer = torch.optim.SGD(model.parameters(), momentum=args.momentum,
                                lr=args.lr, weight_decay=args.weight_decay)
    if args.dataset_name == 'sth_else':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.dataset_name == 'charades':
        criterion = torch.nn.BCEWithLogitsLoss()
    if args.evaluate:
        validate(val_loader, model, criterion,
                 class_to_idx=dataset_val.classes_dict)
        return

    # training, start a logger
    print('training, start a logger')
    tb_logdir = os.path.join(args.logdir, args.logname)
    if not (os.path.exists(tb_logdir)):
        os.makedirs(tb_logdir)
    tb_logger = Logger(tb_logdir)

    if not (os.path.exists(args.ckpt)):
        os.makedirs(args.ckpt)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, criterion, tb_logger)

        # evaluate on validation set
        if (not args.fine_tune) or (epoch + 1) % 10 == 0:
            eval_out = validate(val_loader, model, criterion,
                                epoch=epoch, tb_logger=tb_logger)
            if args.dataset_name == 'sth_else':
                loss, acc_top1 = eval_out
            else:
                loss, acc_top1, mAP = eval_out
        else:
            loss, acc_top1 = 100, 0

        if args.dataset_name == 'sth_else':
            is_best = acc_top1 > best_acc_top1
            best_acc_top1 = max(acc_top1, best_acc_top1)
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict(),
                    'best_acc_top1': best_acc_top1,
                },
                is_best,
                os.path.join(args.ckpt, '{}'.format(args.logname)))
        elif args.dataset_name == 'charades':
            # remember best map
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            save_checkpoint(
                {
                    'epoch': epoch+1,
                    'state_dict': model.module.state_dict(),
                    'best_mAP': best_mAP,
                },
                is_best,
                os.path.join(args.ckpt, '{}'.format(args.logname)))


def train(train_loader, model, optimizer, epoch, criterion, tb_logger=None):
    global args
    batch_time, data_time, losses1, losses2, acc_top1, acc_top5 = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    preds, targets = [], []

    # switch to train mode
    model.train()

    end = time.time()
    for i, (global_img_tensors, box_tensors, box_categories, video_label) in enumerate(train_loader):
        # print('load data...', time.time()-end)
        model.zero_grad()
        # measure data loading time
        data_time.update(time.time() - end)

        vl = video_label.float() if type(
            criterion).__name__ == 'BCEWithLogitsLoss' else video_label.long()
        vl = vl.cuda() if torch.cuda.is_available() else vl

        outputs = model(global_img_tensors, box_categories,
                        box_tensors, video_label)
        if args.pred:
            output, pred_loss = outputs
        else:
            output = outputs

        output = output.view((-1, len(train_loader.dataset.classes)))
        loss = criterion(output, vl)
        pred = torch.sigmoid(output)
        preds.extend(list(pred.cpu().detach().numpy()))
        targets.extend(list(vl.cpu().detach().numpy()))

        loss1, loss2 = loss, loss
        if args.pred:
            loss2 = args.pred_w*pred_loss.mean(0)
            loss += loss2
            # loss = loss2

        # print('model forward...', time.time()-since)
        acc1, acc5 = accuracy(output.cpu(), video_label, topk=(1, 5))

        # measure accuracy and record loss
        batch_size = box_tensors.size(0)
        losses1.update(loss1.item(), batch_size)
        losses2.update(loss2.item(), batch_size)
        acc_top1.update(acc1.item(), batch_size)
        acc_top5.update(acc5.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.clip_gradient is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.clip_gradient)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if args.dataset_name == 'charades':
        mAP, _, AP = charades_map(np.stack(preds), np.stack(targets))
        print('Epoch[{0}](Train):\t'
              'Time {batch_time.sum:.3f}\t'
              'Data {data_time.sum:.3f}\t'
              'Loss1 {loss1.avg:.4f}\t'
              'Loss2 {loss2.avg:.4f}\t'
              'Acc1 {acc_top1.avg:.1f}\t'
              'Acc5 {acc_top5.avg:.1f}\t'
              'mAP {mAP:.1f}\t'.format(epoch, batch_time=batch_time, data_time=data_time,
                                       loss1=losses1, loss2=losses2, acc_top1=acc_top1, acc_top5=acc_top5, mAP=mAP))
    else:
        print('Epoch[{0}](Train):\t'
              'Time {batch_time.sum:.3f}\t'
              'Data {data_time.sum:.3f}\t'
              'Loss1 {loss1.avg:.4f}\t'
              'Loss2 {loss2.avg:.4f}\t'
              'Acc1 {acc_top1.avg:.1f}\t'
              'Acc5 {acc_top5.avg:.1f}'.format(epoch, batch_time=batch_time, data_time=data_time, loss1=losses1, loss2=losses2, acc_top1=acc_top1, acc_top5=acc_top5))


def validate(val_loader, model, criterion, epoch=None, tb_logger=None, class_to_idx=None):
    batch_time, data_time, losses1, losses2, acc_top1, acc_top5 = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    logits_matrix, targets_list, preds, targets = [], [], [], []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (global_img_tensors, box_tensors, box_categories, video_label) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # compute output

        vl = video_label.float() if type(
            criterion).__name__ == 'BCEWithLogitsLoss' else video_label.long()
        vl = vl.cuda() if torch.cuda.is_available() else vl

        with torch.no_grad():
            outputs = model(global_img_tensors, box_categories,
                            box_tensors, video_label, is_inference=True)
            if args.pred:
                output, pred_loss = outputs
            else:
                output = outputs

            output = output.view((-1, len(val_loader.dataset.classes)))
            loss = criterion(output, vl)
            pred = torch.sigmoid(output)
            preds.extend(list(pred.cpu().numpy()))
            targets.extend(list(vl.cpu().numpy()))

            loss1, loss2 = loss, loss
            if args.pred:
                loss2 = args.pred_w*pred_loss.mean(0)
                loss += loss2
                # loss = loss2

            acc1, acc5 = accuracy(output.cpu(), video_label, topk=(1, 5))

            if args.evaluate:
                logits_matrix.append(output.cpu().data.numpy())
                targets_list.append(video_label.cpu().numpy())
                # print('logits_matrix mem_size: ', sys.getsizeof(logits_matrix))
                # print('targets_list mem_size: ', sys.getsizeof(targets_list))

        # measure accuracy and record loss
        batch_size = box_tensors.size(0)
        losses1.update(loss1.item(), batch_size)
        losses2.update(loss2.item(), batch_size)
        acc_top1.update(acc1.item(), batch_size)
        acc_top5.update(acc5.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if args.dataset_name == 'charades':
        mAP, _, AP = charades_map(np.stack(preds), np.stack(targets))
        # print(mAP, ap)
        print('Epoch[{0}](Test):\t'
              'Time {batch_time.sum:.3f}\t'
              'Data {data_time.sum:.3f}\t'
              'Loss1 {loss1.avg:.4f}\t'
              'Loss2 {loss2.avg:.4f}\t'
              'Acc1 {acc_top1.avg:.1f}\t'
              'Acc5 {acc_top5.avg:.1f}\t'
              'mAP {mAP:.1f}\t'.format(epoch, batch_time=batch_time, data_time=data_time,
                                       loss1=losses1, loss2=losses2, acc_top1=acc_top1, acc_top5=acc_top5, mAP=mAP))
        print('#'*60)

    else:
        print('Epoch[{0}](Test):\t'
              'Time {batch_time.sum:.3f}\t'
              'Data {data_time.sum:.3f}\t'
              'Loss1 {loss1.avg:.4f}\t'
              'Loss2 {loss2.avg:.4f}\t'
              'Acc1 {acc_top1.avg:.1f}\t'
              'Acc5 {acc_top5.avg:.1f}'.format(epoch, batch_time=batch_time, data_time=data_time,
                                               loss1=losses1, loss2=losses2, acc_top1=acc_top1, acc_top5=acc_top5))
        print('#'*60)

    if args.evaluate:
        logits_matrix = np.concatenate(logits_matrix)
        targets_list = np.concatenate(targets_list)
        save_results(logits_matrix, targets_list, class_to_idx, args)

    if args.dataset_name == 'charades':
        return losses1.avg, acc_top1.avg, mAP
    return losses1.avg, acc_top1.avg


if __name__ == '__main__':
    main()
