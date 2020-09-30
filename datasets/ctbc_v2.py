import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image

from .generic import DenseDataset, SparseDataset, VideoRecord

DATASET_ROOT = R"G:\Datasets\CTBCYogie2\cropped_frames"


class CTBC_Sparse(SparseDataset):
    def __init__(self, root='', train=True, static=True, which_cam=1, num_segments=16, which_split=1, transform=None, **kwargs):
        super().__init__(num_segments)
        self.train = train
        self.static = static
        self.which_cam = which_cam
        self.which_split = which_split
        self.transform = transform

        self.root = root
        if not self.root:
            self.root = DATASET_ROOT
            self.root += "_static" if self.static else "_dynamic"

        self.video_list = []
        self._parse_list()

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        record = self.video_list[idx]

        if self.train:
            indices = self._get_train_indices(record.num_frames)
        else:
            indices = self._get_test_indices(record.num_frames)
        
        return self._load_video(record, indices), record.label

    def _load_video(self, record, indices):
        image_list = sorted(os.listdir(record.path))

        imgs = []
        for seg_idx in indices:
            seg_idx = int(seg_idx)
            img = torch.from_numpy(np.array(Image.open(os.path.join(record.path, image_list[seg_idx]))))
            imgs.append(img)

        imgs = torch.stack(imgs, dim=0)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs

    def _parse_list(self):
        self._parse_class_names()

        annot_path = os.path.join(os.path.dirname(__file__), 'lists/ctbc_v2/ctbc_v2_%s_split%02d.csv' % ('train' if self.train else 'test', self.which_split))
        video_info = pd.read_csv(annot_path, header=None)

        for _, row in video_info.iterrows():
            vpath, vsubject, vcam, vlabel, vlen = row
            if self.which_cam == 0 or (int(vcam) == self.which_cam):
                self.video_list.append(VideoRecord((os.path.join(self.root, vpath), vlen, vlabel)))

    def _parse_class_names(self):
        self.class_dict_encode = {}
        self.class_dict_decode = {}

        class_desc_file = os.path.join(os.path.dirname(__file__), 'lists/ctbc_v2/ctbc_classInd.txt')
        class_info = pd.read_csv(class_desc_file, sep=' ', header=None)
        for _, row in class_info.iterrows():
            class_idx, class_name = row
            self.class_dict_decode[class_idx] = class_name
            self.class_dict_encode[class_name] = class_idx

    def encode_class(self, class_name):
        return self.class_dict_encode[class_name]

    def decode_class(self, class_idx):
        return self.class_dict_decode[class_idx]
        

class CTBC_Dense(DenseDataset):
    def __init__(self, root='', train=True, static=True, which_cam=1, num_segments=10, which_split=1, transform=None, num_frames_per_clip=16, sample_uniform=False, temporal_stride=1, **kwargs):
        super().__init__(num_segments, num_frames_per_clip, temporal_stride)
        self.train = train
        self.static = static
        self.which_cam = which_cam
        self.which_split = which_split
        self.transform = transform
        self.sample_uniform = sample_uniform

        self.root = root
        if not self.root:
            self.root = DATASET_ROOT
            self.root += "_static" if self.static else "_dynamic"

        self.video_list = []
        self._parse_list()

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        record = self.video_list[idx]

        if self.sample_uniform:
            indices = self._get_uniform_indices(record.num_frames)
        else:
            indices = self._get_random_indices(record.num_frames)
        
        return self._load_video(record, indices), record.label

    def _load_video(self, record, indices):
        image_list = sorted(os.listdir(record.path))
        clips = []
        for seg_idx in indices:
            seg_idx = int(seg_idx)
            clip = self._load_clip(record, image_list, seg_idx)

            if self.transform is not None:
                clip = self.transform(clip)
            
            clips.append(clip)

        return torch.stack(clips, dim=0)

    def _load_clip(self, record, image_list, start_idx):
        seg_indices = np.arange(start_idx, start_idx + self.clip_size, self.temporal_stride, dtype=np.int)

        imgs = []
        for idx in seg_indices:
            if idx >= len(image_list):
                idx = len(image_list) - 1
            img = torch.from_numpy(np.array(Image.open(os.path.join(record.path, image_list[idx]))))
            imgs.append(img)

        return torch.stack(imgs, dim=0)

    def _parse_list(self):
        self._parse_class_names()

        annot_path = os.path.join(os.path.dirname(__file__), 'lists/ctbc_v2/ctbc_v2_%s_split%02d.csv' % ('train' if self.train else 'test', self.which_split))
        video_info = pd.read_csv(annot_path, header=None)

        for _, row in video_info.iterrows():
            vpath, vsubject, vcam, vlabel, vlen = row
            if self.which_cam == 0 or (int(vcam) == self.which_cam):
                self.video_list.append(VideoRecord((os.path.join(self.root, vpath), vlen, vlabel)))

    def _parse_class_names(self):
        self.class_dict_encode = {}
        self.class_dict_decode = {}

        class_desc_file = os.path.join(os.path.dirname(__file__), 'lists/ctbc_v2/ctbc_classInd.txt')
        class_info = pd.read_csv(class_desc_file, sep=' ', header=None)
        for _, row in class_info.iterrows():
            class_idx, class_name = row
            self.class_dict_decode[class_idx] = class_name
            self.class_dict_encode[class_name] = class_idx

    def encode_class(self, class_name):
        return self.class_dict_encode[class_name]

    def decode_class(self, class_idx):
        return self.class_dict_decode[class_idx]