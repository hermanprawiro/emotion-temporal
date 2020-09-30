import glob
import os
import random

import numpy as np
import torch
from PIL import Image


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class SparseDataset(torch.utils.data.Dataset):
    def __init__(self, num_segments):
        self.num_segments = num_segments

    def _get_train_indices(self, num_frames):
        average_duration = num_frames // self.num_segments
        if average_duration > 0:
            indices = np.multiply(list(range(self.num_segments)), average_duration) + np.random.randint(average_duration, size=self.num_segments)
        elif num_frames > self.num_segments:
            indices = np.sort(np.random.randint(num_frames, size=self.num_segments))
        else:
            indices = np.zeros((self.num_segments,))

        return indices

    def _get_test_indices(self, num_frames):
        average_duration = num_frames / float(self.num_segments)
        indices = np.array([int(average_duration / 2.0 + average_duration * x) for x in range(self.num_segments)])

        return indices

    def _get_val_indices(self, num_frames):
        if num_frames > self.num_segments:
            indices = self._get_test_indices(num_frames)
        else:
            indices = np.zeros((self.num_segments,))

        return indices


class DenseDataset(torch.utils.data.Dataset):
    def __init__(self, num_segments, num_frames_per_clip, temporal_stride):
        self.num_segments = num_segments
        self.num_frames_per_clip = num_frames_per_clip
        self.temporal_stride = temporal_stride
        self.clip_size = self.temporal_stride * (self.num_frames_per_clip - 1) + 1
            
    def _get_random_indices(self, num_frames):
        end_index = num_frames - self.clip_size

        if end_index > 0:
            indices = np.sort(np.random.choice(end_index, size=self.num_segments, replace=False))
        else:
            indices = np.zeros((self.num_segments,))

        return indices

    def _get_uniform_indices(self, num_frames):
        average_duration = num_frames / float(self.num_segments)
        end_index = num_frames - self.clip_size

        if average_duration > self.clip_size:
            indices = np.array([int(average_duration / 2.0 + average_duration * x - (self.clip_size / 2)) for x in range(self.num_segments)])
        elif end_index > 0:
            indices = np.linspace(0, end_index-1, self.num_segments, dtype=np.int)
        else:
            indices = np.zeros((self.num_segments,))

        return indices


class SimpleImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform

        if not os.path.isdir(self.root):
            raise NotADirectoryError("The root is not a directory or does not exist.")

        self.images_list = []
        self._parse_list()

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img = Image.open(self.images_list[idx])

        if self.transform is not None:
            img = self.transform(img)

        return img

    def _parse_list(self):
        self.images_list = sorted(glob.glob(os.path.join(self.root, "*.jpg")))


class SimpleSparseDataset(SparseDataset):
    def __init__(self, root, num_clips=1, num_segments=16, sample_uniform=True, transform=None):
        super().__init__(num_segments)
        self.root = root
        self.transform = transform
        self.num_clips = num_clips
        self.sample_uniform = sample_uniform

        if not os.path.isdir(self.root):
            raise NotADirectoryError("The root is not a directory or does not exist.")

        self.images_list = []
        self._parse_list()

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        num_frames = len(self.images_list)

        if self.sample_uniform:
            indices = self._get_test_indices(num_frames)
        else:
            indices = self._get_train_indices(num_frames)
        
        return self._load_video(indices)

    def _load_video(self, indices):
        imgs = []
        for seg_idx in indices:
            seg_idx = int(seg_idx)
            img = torch.from_numpy(np.array(Image.open(self.images_list[seg_idx])))
            imgs.append(img)

        imgs = torch.stack(imgs, dim=0)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs

    def _parse_list(self):
        self.images_list = sorted(glob.glob(os.path.join(self.root, "*.jpg")))
