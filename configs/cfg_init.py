import argparse
from utils import str2bool


def main_args():
    parser = argparse.ArgumentParser(description='PyTorch Smth-Else')
    parser.add_argument(
        '--dataset_name', choices=['sth_else', 'charades'], default='sth_else')
    parser.add_argument(
        '--dataset_root', default='dataset')
    parser.add_argument('--model', default='coord')
    args, _ = parser.parse_known_args()

    parser.add_argument('--root_frames', default='%s/%s/frames' %
                        (args.dataset_root, args.dataset_name), type=str, help='path to the folder with frames')
    parser.add_argument('--json_data_train', default='dataset_splits/%s/compositional/train.json' %
                        (args.dataset_name), type=str, help='path to the json file with train video meta data')
    parser.add_argument('--json_data_val', default='dataset_splits/%s/compositional/validation.json' %
                        (args.dataset_name), type=str, help='path to the json file with validation video meta data')
    parser.add_argument('--json_file_labels', default='dataset_splits/%s/compositional/labels.json' %
                        (args.dataset_name), type=str, help='path to the json file with ground truth labels')
    parser.add_argument('--tracked_boxes', default='dataset/%s/bounding_box_annotations.pkl' %
                        (args.dataset_name), type=str, help='choose tracked boxes')

    parser.add_argument('--img_feature_dim', default=256, type=int, metavar='N',
                        help='intermediate feature dimension for image-based features')
    parser.add_argument('--coord_feature_dim', default=512, type=int, metavar='N',
                        help='intermediate feature dimension for coord-based features')
    parser.add_argument('--clip_gradient', '-cg', default=5, type=float,
                        metavar='W', help='gradient norm clipping (default: 5)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--size', default=224, type=int, metavar='N',
                        help='primary image input size')
    parser.add_argument('--start_epoch', default=None, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', '-b', default=72, type=int,
                        metavar='N', help='mini-batch size (default: 72)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_steps', default=[24, 35, 45], type=float, nargs="+",
                        metavar='LRSteps', help='epochs to decay learning rate by 10')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=0.0001, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print_freq', '-p', default=20, type=int,
                        metavar='N', help='print frequency (default: 20)')
    parser.add_argument('--log_freq', '-l', default=10, type=int,
                        metavar='N', help='frequency to write in tensorboard (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--num_classes', default=174, type=int,
                        help='num of class in the model')
    parser.add_argument('--num_boxes', default=4, type=int,
                        help='num of boxes for each image')
    parser.add_argument('--num_frames', default=8, type=int,
                        help='num of frames for the model')  # it is very important
    parser.add_argument('--dataset', default='smth_smth',
                        help='which dataset to train')
    parser.add_argument('--logdir', default='./logs',
                        help='folder to output tensorboard logs')
    parser.add_argument('--logname', default='exp',
                        help='name of the experiment for checkpoints and logs')
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--fine_tune', help='path with ckpt to restore')
    parser.add_argument('--shot', default=0)
    parser.add_argument('--restore_i3d')
    parser.add_argument('--restore_custom')

    # added by Mr. Yan
    parser.add_argument('--if_augment', type=str2bool,
                        nargs='?', const=True, default=False)
    parser.add_argument('--pred', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--pred_w', default=1.0, type=float)
    args, _ = parser.parse_known_args()
    return args


def model_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--box_mode', type=str, choices=['GT', 'DET'], default='GT')
    parser.add_argument('--GLOBAL', type=str2bool,
                        nargs='?', const=True, default=False)
    parser.add_argument('--vis_info', type=str2bool,
                        nargs='?', const=True, default=True)
    parser.add_argument('--coord_info', type=str2bool,
                        nargs='?', const=True, default=False)
    parser.add_argument('--category_info', type=str2bool,
                        nargs='?', const=True, default=False)
    parser.add_argument('--i3d', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--reasoning_module', type=str, choices=[
                        'pool', 'pool_T', 'STIN', 'STNL', 'STCR', 'STRG'], default='pool')
    parser.add_argument('--reasoning_mode', type=str,
                        choices=['ST', 'T'], default='ST')
    parser.add_argument('--hidden_feature_dim', type=int, default=512)
    parser.add_argument('--joint', type=str2bool,
                        nargs='?', const=True, default=False)
    parser.add_argument('--obs_len_ratio', type=float, default=0.5)
    parser.add_argument('--LSTM_flow', type=str2bool,
                        nargs='?', const=True, default=False)
    parser.add_argument('--multiple_interaction', type=str2bool,
                        nargs='?', const=True, default=False)
    args, _ = parser.parse_known_args()
    return args


def data_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle_order', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--vis_info', type=str2bool, nargs='?',
                        const=True, default=False)
    args, _ = parser.parse_known_args()
    return args
