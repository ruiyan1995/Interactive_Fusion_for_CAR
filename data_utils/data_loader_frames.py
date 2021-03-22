import os
from os.path import join
from torchvision.transforms import Compose
import numpy as np
from PIL import Image
import torch
from data_utils import gtransforms
from data_utils.data_parser import WebmDataset
from numpy.random import choice as ch
import json
import time
import cv2
import random
from configs import cfg_init

args = cfg_init.data_args()


class VideoFolder(torch.utils.data.Dataset):
    """
    Something-Something dataset based on *frames* extraction
    """

    def __init__(self,
                 root,
                 file_input,
                 file_labels,
                 frames_duration,
                 args=None,
                 multi_crop_test=False,
                 sample_rate=2,
                 is_test=False,
                 is_val=False,
                 num_boxes=10,
                 model=None,
                 if_augment=True,
                 anno=None,
                 subset_rate=1
                 ):
        """
        :param root: data root path
        :param file_input: inputs path
        :param file_labels: labels path
        :param frames_duration: number of frames
        :param multi_crop_test:
        :param sample_rate: FPS
        :param is_test: is_test flag
        :param k_split: number of splits of clips from the video
        :param sample_split: how many frames sub-sample from each clip
        """
        self.in_duration = frames_duration
        self.coord_nr_frames = self.in_duration // 2
        self.multi_crop_test = multi_crop_test
        self.sample_rate = sample_rate
        self.if_augment = if_augment
        self.is_val = is_val
        self.data_root = root
        self.dataset_object = WebmDataset(
            args.dataset_name, file_input, file_labels, root, is_test=is_test, subset_rate=subset_rate)
        self.json_data = self.dataset_object.json_data
        self.classes = self.dataset_object.classes
        self.classes_dict = self.dataset_object.classes_dict

        self.model = model
        self.num_boxes = num_boxes

        self.dataset_name = args.dataset_name
        self.num_classes = args.num_classes

        # Prepare data for the data loader
        self.prepare_data()
        self.args = args
        self.pre_resize_shape = (256, 340)
        self.box_annotations = anno

        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]

        # Transformations
        if not self.is_val:
            self.transforms = [
                gtransforms.GroupResize((224, 224)),
            ]
        elif self.multi_crop_test:
            self.transforms = [
                gtransforms.GroupResize((256, 256)),
                gtransforms.GroupRandomCrop((256, 256)),
            ]
        else:
            self.transforms = [
                gtransforms.GroupResize((224, 224))
                # gtransforms.GroupCenterCrop(256),
            ]
        self.transforms += [
            gtransforms.ToTensor(),
            gtransforms.GroupNormalize(self.img_mean, self.img_std),
        ]
        self.transforms = Compose(self.transforms)

        if self.if_augment:
            if not self.is_val:  # train, multi scale cropping
                self.random_crop = gtransforms.GroupMultiScaleCrop(output_size=224,
                                                                   scales=[1, .875, .75])
            else:  # val, only center cropping
                self.random_crop = gtransforms.GroupMultiScaleCrop(output_size=224,
                                                                   scales=[1],
                                                                   max_distort=0,
                                                                   center_crop_only=True)
        else:
            self.random_crop = None

    def prepare_data(self):
        """
        This function creates 3 lists: vid_names, labels and frame_cnts
        :return:
        """
        print("Loading label strings")
        self.label_strs = ['_'.join(class_name.split(' '))
                           for class_name in self.classes]
        vid_names = []
        frame_cnts = []
        frame_ids = []
        labels = []
        since = time.time()

        video_info_json_pth = 'dataset/%s/BBOX/video_info.json' % (
            self.dataset_name)
        if os.path.exists(video_info_json_pth):
            with open(video_info_json_pth, 'r') as fp:
                self.video_info_json = json.load(fp)
        else:
            self.video_info_json = None

        for idx, listdata in enumerate(self.json_data):
            try:
                vid_names.append(listdata.id)
                labels.append(listdata.label)
                if self.video_info_json:
                    frame_cnts.append(self.video_info_json[listdata.id]['cnt'])
                    if self.dataset_name == 'charades':
                        frame_ids.append(
                            self.video_info_json[listdata.id]['frame_ids'])
                else:
                    frames_path = join(os.path.dirname(
                        self.data_root), "frames/{list_id}".format(list_id=listdata.id))
                    frame_cnts.append(len(frames_path))
                    frames = os.listdir(frames_path)
                    if self.dataset_name == 'charades':
                        frames.sort(key=lambda x: int(x[:-4]))
                        ids = []
                        for fr in frames:
                            ids.append(int(fr.split('.')[0]))
                        frame_ids.append(ids)

            except Exception as e:
                print(str(e))

        self.vid_names = vid_names
        self.labels = labels
        self.frame_cnts = frame_cnts
        self.frame_ids = frame_ids

    # todo: might consider to replace it to opencv, should be much faster
    def load_frame(self, vid_name, frame_idx):
        """
        Load frame
        :param vid_name: video name
        :param frame_idx: index
        :return:
        """
        if self.dataset_name == 'sth_else':
            file_path = join(os.path.dirname(self.data_root),
                             'frames', vid_name, '%04d.jpg' % (frame_idx + 1))
        elif self.dataset_name == 'charades':
            file_path = join(os.path.dirname(self.data_root),
                             'frames', vid_name, '%06d.png' % (frame_idx))

        return Image.fromarray(cv2.imread(file_path)).convert('RGB')

    def _sample_indices(self, nr_video_frames):
        # random sampling
        average_duration = nr_video_frames * 1.0 / self.coord_nr_frames
        if average_duration > 0:
            offsets = np.multiply(list(range(self.coord_nr_frames)), average_duration) \
                + np.random.uniform(0, average_duration,
                                    size=self.coord_nr_frames)
            offsets = np.floor(offsets)
        elif nr_video_frames > self.coord_nr_frames:
            offsets = np.sort(np.random.randint(
                nr_video_frames, size=self.coord_nr_frames))
        else:
            offsets = np.zeros((self.coord_nr_frames,))
        offsets = list(map(int, list(offsets)))
        return offsets

    def _get_val_indices(self, nr_video_frames):
        # fixed sampling
        if nr_video_frames > self.coord_nr_frames:
            tick = nr_video_frames * 1.0 / self.coord_nr_frames
            offsets = np.array([int(tick / 2.0 + tick * x)
                                for x in range(self.coord_nr_frames)])
        else:
            offsets = np.zeros((self.coord_nr_frames,))
        offsets = list(map(int, list(offsets)))
        return offsets

    def sample_single(self, index):
        """
        Choose and Load frames per video
        :param index:
        :return:
        """
        n_frame = self.frame_cnts[index] - 1
        d = self.in_duration * self.sample_rate  # 16 * 2
        if n_frame > d:
            if not self.is_val:
                # random sample
                offset = np.random.randint(0, n_frame - d)
            else:
                # center crop
                offset = (n_frame - d) // 2
            frame_list = list(range(offset, offset + d, self.sample_rate))
        else:
            # Temporal Augmentation
            if not self.is_val:  # train
                if n_frame - 2 < self.in_duration:
                    # less frames than needed
                    pos = np.linspace(0, n_frame - 2, self.in_duration)
                else:  # take one
                    pos = np.sort(np.random.choice(
                        list(range(n_frame - 2)), self.in_duration, replace=False))
            else:
                pos = np.linspace(0, n_frame - 2, self.in_duration)
            frame_list = [round(p) for p in pos]

        frame_list = [int(x) for x in frame_list]

        if not self.is_val:  # train
            coord_frame_list = self._sample_indices(n_frame)
        else:  # val
            coord_frame_list = self._get_val_indices(n_frame)

        assert len(coord_frame_list) == len(frame_list) // 2

        if self.model.startswith('region'):
            frame_list = coord_frame_list

        if args.shuffle_order and not self.is_val:
            random.shuffle(frame_list)

        folder_id = str(self.vid_names[index])
        video_data = self.box_annotations[folder_id]

        # union the objects of two frames
        object_set = set()
        for frame_id in coord_frame_list:
            try:
                frame_data = video_data[frame_id]
            except:
                frame_data = {'labels': []}
            for box_data in frame_data['labels']:
                standard_category = box_data['standard_category']
                object_set.add(standard_category)
        object_set = sorted(list(object_set))

        ################ for loading, resizing, and cropping frames ####################
        frames = []
        if self.model.startswith('coord') or (self.model.startswith('region') and not args.vis_info):
            pass
            # frames.append(self.load_frame(self.vid_names[index], 0)) # load a frame randomly
        else:
            for fidx in frame_list:
                if self.dataset_name == 'charades':
                    fidx = self.frame_ids[index][fidx]
                frames.append(self.load_frame(self.vid_names[index], fidx))
            # for test
            # frames.append(self.load_frame(self.vid_names[index], 0)) # load a frame randomly

        #height, width = frames[0].height, frames[0].width
        height, width = self.video_info_json[self.vid_names[index]]['res']
        if frames:
            frames = [img.resize(
                (self.pre_resize_shape[1], self.pre_resize_shape[0]), Image.BILINEAR) for img in frames]

        if self.random_crop is not None:
            if frames:
                frames, (offset_h, offset_w, crop_h,
                         crop_w) = self.random_crop(img_group=frames)
            else:
                (offset_h, offset_w, crop_h, crop_w) = self.random_crop(
                    im_size=(width, height))
        else:
            offset_h, offset_w, (crop_h, crop_w) = 0, 0, self.pre_resize_shape
        ################################ modified by Mr. Yan ###############################

        scale_resize_w, scale_resize_h = self.pre_resize_shape[1] / float(
            width), self.pre_resize_shape[0] / float(height)
        scale_crop_w, scale_crop_h = 224 / float(crop_w), 224 / float(crop_h)

        box_tensors = torch.zeros(
            (self.coord_nr_frames, self.num_boxes, 4), dtype=torch.float32)  # (cx, cy, w, h)
        box_categories = torch.zeros((self.coord_nr_frames, self.num_boxes))
        for frame_index, frame_id in enumerate(coord_frame_list):
            try:
                frame_data = video_data[frame_id]
            except:
                frame_data = {'labels': []}
            for box_data in frame_data['labels']:
                standard_category = box_data['standard_category']
                global_box_id = object_set.index(standard_category)

                box_coord = box_data['box2d']
                x0, y0, x1, y1 = box_coord['x1'], box_coord['y1'], box_coord['x2'], box_coord['y2']

                # scaling due to initial resize
                x0, x1 = x0 * scale_resize_w, x1 * scale_resize_w
                y0, y1 = y0 * scale_resize_h, y1 * scale_resize_h

                # shift
                x0, x1 = x0 - offset_w, x1 - offset_w
                y0, y1 = y0 - offset_h, y1 - offset_h

                x0, x1 = np.clip([x0, x1], a_min=0, a_max=crop_w-1)
                y0, y1 = np.clip([y0, y1], a_min=0, a_max=crop_h-1)

                # scaling due to crop
                x0, x1 = x0 * scale_crop_w, x1 * scale_crop_w
                y0, y1 = y0 * scale_crop_h, y1 * scale_crop_h

                # precaution
                x0, x1 = np.clip([x0, x1], a_min=0, a_max=223)
                y0, y1 = np.clip([y0, y1], a_min=0, a_max=223)

                # (cx, cy, w, h)
                gt_box = np.array(
                    [(x0 + x1) / 2., (y0 + y1) / 2., x1 - x0, y1 - y0], dtype=np.float32)

                # normalize gt_box into [0, 1]
                gt_box /= 224

                # load box into tensor
                try:
                    box_tensors[frame_index, global_box_id] = torch.tensor(
                        gt_box).float()
                except:
                    pass
                # load box category #### need a try ... except ... ? Mr. Yan
                try:
                    # box_categories[frame_index, global_box_id] = 1 if box_data['standard_category'] == 'hand' else 2  # 0 is for none
                    box_categories[frame_index, global_box_id] = 1 if box_data['standard_category'] in [
                        'hand', 'person'] else 2  # 0 is for none
                except:
                    pass
                
                # # load image into tensor
                # x0, y0, x1, y1 = list(map(int, [x0, y0, x1, y1]))

        return frames, box_tensors, box_categories, frame_list

    def __getitem__(self, index):
        frames, box_tensors, box_categories, frame_ids = self.sample_single(
            index)

        if self.model.startswith('coord') or (self.model.startswith('region') and not args.vis_info):
            global_img_tensors = []
        else:
            frames = self.transforms(frames)
            global_img_tensors = frames.permute(1, 0, 2, 3)

        if type(self.labels[index]) == list:
            classes = []
            for label in self.labels[index]:
                classes.append(self.classes_dict[label])
            classes = torch.as_tensor(
                self.as_binary_vector(classes, self.num_classes))
        else:
            classes = self.classes_dict[self.labels[index]]

        return global_img_tensors, box_tensors, box_categories, classes

    def __len__(self):
        return len(self.json_data)

    def unnormalize(self, img, divisor=255):
        """
        The inverse operation of normalization
        Both the input & the output are in the format of BxCxHxW
        """
        for c in range(len(self.img_mean)):
            img[:, c, :, :].mul_(self.img_std[c]).add_(self.img_mean[c])

        return img / divisor

    def as_binary_vector(self, labels, num_classes):
        """
        Construct binary label vector given a list of label indices.
        Args:
            labels (list): The input label list.
            num_classes (int): Number of classes of the label vector.
        Returns:
            labels (numpy array): the resulting binary vector.
        """
        label_arr = np.zeros((num_classes,))

        for lbl in set(labels):
            label_arr[lbl] = 1.0
        return label_arr
